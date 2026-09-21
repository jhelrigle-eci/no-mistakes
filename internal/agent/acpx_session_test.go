//go:build unix

package agent

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// ACP initialize responses, in the shape the real servers answer with. The
// capability under test is agentCapabilities.loadSession; cursor-agent
// advertises it, and a target that does not is what the isolation gate has to
// leave untouched.
const (
	acpxInitLoadSession = `{"jsonrpc":"2.0","id":0,"result":{"protocolVersion":1,` +
		`"agentCapabilities":{"loadSession":true,"promptCapabilities":{"image":true}}}}`
	acpxInitNoLoadSession = `{"jsonrpc":"2.0","id":0,"result":{"protocolVersion":1,` +
		`"agentCapabilities":{"promptCapabilities":{"image":true}}}}`
)

// writeSessionStubAcpx writes a stub acpx that appends every invocation's argv
// to a log and mirrors the ACP JSON-RPC conversation the real acpx relays on
// stdout: the target's initialize response, and - for a `session/new` turn
// only - the session id the target minted. A resumed turn answers without one,
// exactly as ACP's session/load does. The session-management subcommands the
// resume path issues exit silently, like the real ones.
//
// mirrorInitOnPrompt selects which of the two shapes a `prompt --session` turn
// takes. Whichever acpx process opens the ACP connection mirrors initialize,
// and a prompt is served by an already-initialized queue owner that need not
// repeat it, so a prompt turn may legitimately carry no capability line at
// all. Both shapes must keep the identity.
func writeSessionStubAcpx(t *testing.T, dir, initLine, newSessionID string, mirrorInitOnPrompt bool) string {
	t.Helper()
	path := filepath.Join(dir, "acpx")
	initGuard := ""
	if !mirrorInitOnPrompt {
		initGuard = `if [ "$mode" != "prompt" ]; then`
	}
	initEnd := ""
	if initGuard != "" {
		initEnd = "fi"
	}
	script := `#!/bin/sh
printf '%s\n' "$*" >> "$NM_TEST_ACPX_LOG"
mode=""
for a in "$@"; do
	case "$a" in
		exec|prompt|sessions|cancel) if [ -z "$mode" ]; then mode="$a"; fi ;;
	esac
done
if [ "$mode" = "sessions" ] || [ "$mode" = "cancel" ]; then
	exit 0
fi
cat > /dev/null
` + initGuard + `
printf '%s\n' '` + initLine + `'
` + initEnd + `
if [ "$mode" = "exec" ]; then
	printf '%s\n' '{"jsonrpc":"2.0","id":1,"result":{"sessionId":"` + newSessionID + `"}}'
fi
printf '%s\n' '{"method":"session/update","params":{"update":{"sessionUpdate":"agent_message_chunk","text":"ok"}}}'
`
	if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}

// readAcpxInvocations returns one entry per stub acpx invocation, in order.
func readAcpxInvocations(t *testing.T, logPath string) []string {
	t.Helper()
	data, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatalf("stub acpx recorded no invocation: %v", err)
	}
	return strings.Split(strings.TrimRight(string(data), "\n"), "\n")
}

// TestAcpxAgent_AdvertisedLoadSessionIsMintedThenResumedWithSessionLoad proves
// the ACP resume contract end to end against a real acpx spawn: a turn with no
// stored identity runs the one-shot `exec` form, reports the id the target
// minted at session/new, and a later turn handed that id reconnects to it
// rather than opening a new session.
//
// The argv assertions are the observable proof that session/load is what
// happens: acpx owns the ACP conversation, and `sessions ensure
// --resume-session` followed by `prompt --session` is the only acpx form that
// reconnects to an existing ACP session instead of minting one.
func TestAcpxAgent_AdvertisedLoadSessionIsMintedThenResumedWithSessionLoad(t *testing.T) {
	const sessionID = "acp-session-7f3a"

	dir := t.TempDir()
	logPath := filepath.Join(dir, "invocations.log")
	t.Setenv("NM_TEST_ACPX_LOG", logPath)
	a := &acpxAgent{bin: writeSessionStubAcpx(t, dir, acpxInitLoadSession, sessionID, true), target: "cursor"}

	// Turn one has no identity to resume, so it must take the cold path.
	first, err := a.Run(context.Background(), RunOpts{Prompt: "first turn", CWD: dir})
	if err != nil {
		t.Fatalf("first turn: %v", err)
	}
	if first.SessionID != sessionID {
		t.Errorf("first turn SessionID = %q, want the id the target minted (%q)", first.SessionID, sessionID)
	}
	if first.Resumed {
		t.Error("first turn reported Resumed, but it had no identity to resume")
	}
	invocations := readAcpxInvocations(t, logPath)
	if len(invocations) != 1 {
		t.Fatalf("cold turn spawned acpx %d times, want 1: %q", len(invocations), invocations)
	}
	if !strings.HasSuffix(invocations[0], "exec --file -") {
		t.Errorf("cold turn argv = %q, want the one-shot exec form", invocations[0])
	}

	// Turn two carries the stored identity and must reconnect to it.
	second, err := a.Run(context.Background(), RunOpts{
		Prompt:  "second turn",
		CWD:     dir,
		Session: &SessionRef{ID: sessionID, Agent: a.Name()},
	})
	if err != nil {
		t.Fatalf("resumed turn: %v", err)
	}
	if !second.Resumed {
		t.Error("resumed turn did not report Resumed")
	}
	// session/load answers without an id, so the turn re-reports the one it
	// loaded and the stored slot survives the round.
	if second.SessionID != sessionID {
		t.Errorf("resumed turn SessionID = %q, want the loaded id %q", second.SessionID, sessionID)
	}

	resumed := readAcpxInvocations(t, logPath)[1:]
	joined := strings.Join(resumed, "\n")
	if !strings.Contains(joined, "sessions ensure --name "+acpxSessionName(sessionID)+" --resume-session "+sessionID) {
		t.Errorf("resumed turn never bound the stored ACP identity; invocations:\n%s", joined)
	}
	if !strings.Contains(joined, "prompt --session "+acpxSessionName(sessionID)+" --file -") {
		t.Errorf("resumed turn did not prompt through the bound session; invocations:\n%s", joined)
	}
	for _, inv := range resumed {
		if strings.HasSuffix(inv, "exec --file -") {
			t.Errorf("resumed turn fell back to the one-shot exec form: %q", inv)
		}
	}

	// Turn three is the round that decides whether resume actually compounds.
	// A prompt is served by an already-initialized acpx queue owner, so it
	// need not re-mirror initialize and the capability line can be absent
	// from the turn's stdout entirely. Re-asking the advertised-capability
	// question there would report no identity, drop the stored slot, and send
	// the next round cold - so the identity has to survive a turn that never
	// repeats the advertisement.
	a.bin = writeSessionStubAcpx(t, dir, acpxInitLoadSession, sessionID, false)
	third, err := a.Run(context.Background(), RunOpts{
		Prompt:  "third turn",
		CWD:     dir,
		Session: &SessionRef{ID: sessionID, Agent: a.Name()},
	})
	if err != nil {
		t.Fatalf("third turn: %v", err)
	}
	if !third.Resumed {
		t.Error("third turn did not report Resumed")
	}
	if third.SessionID != sessionID {
		t.Errorf("third turn SessionID = %q, want the loaded id %q kept even though the prompt turn never re-mirrored initialize", third.SessionID, sessionID)
	}
}

// TestAcpxAgent_TargetWithoutLoadSessionRecordsNoIdentity proves the isolation
// gate. acpx serves every ACP target, so the adapter answers SupportsSessionResume
// for the transport; what keeps an individual target untouched is that a target
// which never advertises loadSession never has an identity recorded, so nothing
// upstream can ever ask to resume it and its turns keep the one-shot exec shape
// they have always had. There is no target allowlist doing this work.
func TestAcpxAgent_TargetWithoutLoadSessionRecordsNoIdentity(t *testing.T) {
	dir := t.TempDir()
	logPath := filepath.Join(dir, "invocations.log")
	t.Setenv("NM_TEST_ACPX_LOG", logPath)
	a := &acpxAgent{bin: writeSessionStubAcpx(t, dir, acpxInitNoLoadSession, "unadvertised-id", true), target: "gemini"}

	res, err := a.Run(context.Background(), RunOpts{Prompt: "turn", CWD: dir})
	if err != nil {
		t.Fatalf("turn: %v", err)
	}
	if res.SessionID != "" {
		t.Errorf("SessionID = %q, want none recorded: the target never advertised loadSession", res.SessionID)
	}
	if res.Resumed {
		t.Error("turn reported Resumed without any advertised session support")
	}
	invocations := readAcpxInvocations(t, logPath)
	if len(invocations) != 1 || !strings.HasSuffix(invocations[0], "exec --file -") {
		t.Errorf("invocations = %q, want exactly the unchanged one-shot exec turn", invocations)
	}
}
