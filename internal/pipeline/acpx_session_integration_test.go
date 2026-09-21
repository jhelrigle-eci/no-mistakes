//go:build unix

package pipeline

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/kunchenguid/no-mistakes/internal/agent"
)

// writeSessionStubAcpx materializes a stub acpx for the pipeline package's
// adapter-integration test. It appends every invocation's argv to a log and
// mirrors the ACP JSON-RPC conversation the real acpx relays on stdout: an
// initialize response advertising loadSession, and - for a `session/new` turn
// only - the id the target minted. A resumed turn answers without one, as
// ACP's session/load does.
func writeSessionStubAcpx(t *testing.T, dir, sessionID string) string {
	t.Helper()
	path := filepath.Join(dir, "acpx")
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
printf '%s\n' '{"jsonrpc":"2.0","id":0,"result":{"protocolVersion":1,"agentCapabilities":{"loadSession":true}}}'
if [ "$mode" = "exec" ]; then
	printf '%s\n' '{"jsonrpc":"2.0","id":1,"result":{"sessionId":"` + sessionID + `"}}'
fi
printf '%s\n' '{"method":"session/update","params":{"update":{"sessionUpdate":"agent_message_chunk","text":"ok"}}}'
`
	if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
		t.Fatalf("write stub acpx: %v", err)
	}
	return path
}

// TestAcpxSessions_ReviewTurnsStayColdAndOnlyFixerResumes drives the real
// acpxAgent (backed by a stub acpx) through the two production seams a review
// step actually uses - RunAgent for every review turn, RunAgentSession for the
// fixer - to pin the policy split the ACP resume capability ships under.
//
// The load-bearing case is the third turn: a review that runs AFTER the fixer
// has already stored an identity must still open a new ACP session. Resuming
// there would seat the prescriber of the fixes under review as their certifier,
// which is why review turns bypass RunSessions entirely. Only the fixer, which
// certifies nothing, reconnects - and only because the target advertised
// loadSession.
func TestAcpxSessions_ReviewTurnsStayColdAndOnlyFixerResumes(t *testing.T) {
	const sessionID = "acp-session-c41d"

	dir := t.TempDir()
	logPath := filepath.Join(dir, "invocations.log")
	t.Setenv("NM_TEST_ACPX_LOG", logPath)

	d, run := sessionTestDB(t)
	ag, err := agent.New("acp:cursor", writeSessionStubAcpx(t, dir, sessionID), nil)
	if err != nil {
		t.Fatalf("new acpx agent: %v", err)
	}
	sctx := &StepContext{
		Ctx:      context.Background(),
		Agent:    ag,
		Sessions: NewRunSessions(d, run.ID, ag, true),
	}
	opts := agent.RunOpts{Prompt: "turn", CWD: dir}

	// invocationsSince returns the argv lines the stub recorded after mark,
	// and the new mark.
	invocationsSince := func(mark int) ([]string, int) {
		t.Helper()
		data, err := os.ReadFile(logPath)
		if err != nil {
			t.Fatalf("stub acpx recorded no invocation: %v", err)
		}
		all := strings.Split(strings.TrimRight(string(data), "\n"), "\n")
		return all[mark:], len(all)
	}
	storedSession := func() string {
		t.Helper()
		sessions, err := d.GetRunAgentSessions(run.ID)
		if err != nil {
			t.Fatalf("get run sessions: %v", err)
		}
		if len(sessions) == 0 {
			return ""
		}
		if len(sessions) != 1 || sessions[0].Role != string(SessionRoleFixer) {
			t.Fatalf("stored session rows = %+v, want at most one fixer row", sessions)
		}
		return sessions[0].SessionID
	}
	assertCold := func(what string, invocations []string) {
		t.Helper()
		if len(invocations) != 1 {
			t.Fatalf("%s spawned acpx %d times, want exactly 1 cold turn: %q", what, len(invocations), invocations)
		}
		if !strings.HasSuffix(invocations[0], "exec --file -") {
			t.Errorf("%s argv = %q, want the one-shot exec form", what, invocations[0])
		}
	}

	mark := 0

	// Turn 1: the initial review. Review turns never touch RunSessions.
	if _, err := sctx.RunAgent(opts); err != nil {
		t.Fatalf("initial review turn: %v", err)
	}
	invocations, mark := invocationsSince(mark)
	assertCold("initial review turn", invocations)
	if got := storedSession(); got != "" {
		t.Fatalf("review turn stored session %q; review turns must record no identity", got)
	}

	// Turn 2: the first fixer turn. Nothing is stored yet, so it is cold too,
	// and it records the identity the target minted.
	if _, err := sctx.RunAgentSession(SessionRoleFixer, opts); err != nil {
		t.Fatalf("first fixer turn: %v", err)
	}
	invocations, mark = invocationsSince(mark)
	assertCold("first fixer turn", invocations)
	if got := storedSession(); got != sessionID {
		t.Fatalf("first fixer turn persisted session %q, want %q", got, sessionID)
	}

	// Turn 3: the rereview, with a populated fixer slot. This is the
	// regression: it must still open a new session.
	if _, err := sctx.RunAgent(opts); err != nil {
		t.Fatalf("rereview turn: %v", err)
	}
	invocations, mark = invocationsSince(mark)
	assertCold("rereview turn with a stored fixer session", invocations)
	for _, inv := range invocations {
		if strings.Contains(inv, "--resume-session") || strings.Contains(inv, "prompt --session") {
			t.Fatalf("rereview turn resumed a stored session: %q", inv)
		}
	}

	// Turn 4: the second fixer turn resumes the identity turn 2 stored.
	fix, err := sctx.RunAgentSession(SessionRoleFixer, opts)
	if err != nil {
		t.Fatalf("second fixer turn: %v", err)
	}
	if !fix.Resumed {
		t.Error("second fixer turn did not report Resumed")
	}
	invocations, _ = invocationsSince(mark)
	joined := strings.Join(invocations, "\n")
	if !strings.Contains(joined, "--resume-session "+sessionID) {
		t.Errorf("second fixer turn never bound the stored ACP identity; invocations:\n%s", joined)
	}
	if !strings.Contains(joined, "prompt --session ") {
		t.Errorf("second fixer turn did not prompt through the bound session; invocations:\n%s", joined)
	}
	for _, inv := range invocations {
		if strings.HasSuffix(inv, "exec --file -") {
			t.Errorf("second fixer turn fell back to the one-shot exec form: %q", inv)
		}
	}
	if got := storedSession(); got != sessionID {
		t.Errorf("stored session after resume = %q, want the same identity %q", got, sessionID)
	}
}
