package agent

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"

	"github.com/kunchenguid/no-mistakes/internal/shellenv"
)

const acpxScannerMaxTokenSize = 256 * 1024 * 1024

// acpxSessionCleanupTimeout bounds the best-effort commands that release a
// resumed turn's acpx session record. They run after the turn, on a context
// the turn's own cancellation cannot cut short, so they need a bound of their
// own.
const acpxSessionCleanupTimeout = 30 * time.Second

type acpxAgent struct {
	bin        string
	target     string
	rawCommand string
	// model is the harness-neutral model pin resolved by internal/agentcfg.
	// no-mistakes never speaks ACP itself, so acpx's own --model is the only
	// mechanism that reaches the target agent; empty leaves the target on its
	// configured default, exactly as before the common layer existed.
	model string
	// disableProjectSettings is the resolved, trusted-only opt-out. For the omp
	// target it switches the launch to a neutralized `omp acp` command (see
	// ompgate.go); other targets ignore it and are refused by
	// EnsureGateNeutralized when the opt-out is on.
	disableProjectSettings bool
	// overlayOnce guards a single lazy write of the omp neutralization overlay;
	// overlayPath/overlayErr carry its result across resumed invocations.
	overlayOnce sync.Once
	overlayPath string
	overlayErr  error
	subprocessContext
}

func (a *acpxAgent) Name() string { return "acp:" + a.target }

func (a *acpxAgent) ReportsAgentAttempts() bool { return true }

// SupportsSessionResume reports the ACP transport's durable-session
// capability, not any one target's. no-mistakes never speaks ACP itself: acpx
// is the client, and it mirrors the whole JSON-RPC conversation on stdout, so
// a turn reads the target's own `initialize` response for
// agentCapabilities.loadSession and only reports an identity when the target
// advertised it (acpxSessionFacts.resumableSessionID).
//
// That advertised capability is the isolation gate, and deliberately so: a
// target that does not advertise loadSession never gets an identity recorded,
// so nothing ever asks to resume it and its turns keep the one-shot `exec`
// shape they have always had. There is no target allowlist here on purpose -
// the capability generalizes to every ACP target on its own.
func (a *acpxAgent) SupportsSessionResume() bool { return true }

// NeutralizesGateInstructions reports whether this acpx invocation launches its
// target with the target repository's project instructions neutralized. Only
// the omp target under the trusted opt-out (and only its default launch)
// qualifies; see neutralizesOMPGate.
func (a *acpxAgent) NeutralizesGateInstructions() bool {
	return neutralizesOMPGate(a.target, a.rawCommand, a.disableProjectSettings)
}

func (a *acpxAgent) Run(ctx context.Context, opts RunOpts) (*Result, error) {
	return runWithRetry(ctx, a.Name(), opts, claudeMaxRetries, classifyTransient, nil, func() (*Result, error) {
		return a.runOnce(ctx, opts)
	})
}

func (a *acpxAgent) runOnce(ctx context.Context, opts RunOpts) (*Result, error) {
	prompt := opts.Prompt
	if len(opts.JSONSchema) > 0 {
		prompt = buildACPStructuredPrompt(prompt, opts.JSONSchema)
	}
	rawCommand, err := a.resolveRawCommand()
	if err != nil {
		return nil, fmt.Errorf("acpx omp gate neutralization: %w", err)
	}
	resumeID := ""
	if opts.Session != nil {
		resumeID = opts.Session.ID
	}
	turn := acpxExecTurn()
	if resumeID != "" {
		name := acpxSessionName(resumeID)
		if err := a.loadSession(ctx, rawCommand, opts, name, resumeID); err != nil {
			return nil, err
		}
		defer a.releaseSession(ctx, rawCommand, opts, name)
		turn = acpxPromptTurn(name)
	}
	args := a.buildArgs(rawCommand, opts, turn)
	cmd := exec.CommandContext(ctx, a.bin, args...)
	cmd.Dir = opts.CWD
	cmd.Env = a.gitSafeEnv(opts.CWD, opts.Env)
	shellenv.ConfigureShellCommand(cmd)

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("acpx stdin pipe: %w", err)
	}
	started, err := startNativeAgentCommand(cmd, nativeAgentActivityObserver(opts, a.Name()))
	if err != nil {
		_ = stdin.Close()
		return nil, fmt.Errorf("acpx start: %w", err)
	}
	defer started.closePipes()
	pid := started.pid()
	emitAgentStarted(opts, a.Name(), pid)

	stdinErrCh := writeNativeAgentStdin(stdin, prompt)

	var stderrBuf []byte
	var stderrWG sync.WaitGroup
	stderrWG.Add(1)
	go func() {
		defer stderrWG.Done()
		stderrBuf, _ = io.ReadAll(started.stderr)
	}()

	var usage TokenUsage
	var facts acpxSessionFacts
	text, stdoutErr, err := parseAcpxJSONEvents(ctx, started.stdout, opts.OnChunk, &usage, &facts)
	// Estimate before any return, not just the success one: acpx can report an
	// input-only usage event and then fail, and a reported usage with no output
	// count would otherwise record the text it did stream as a reported zero.
	if usage.OutputTokens == 0 {
		usage.OutputTokens = estimateAcpxTokens(len(text))
	}
	if err != nil {
		err = started.waitAfterParseError(err)
		stderrWG.Wait()
		err = errors.Join(err, acpxStdinError(<-stdinErrCh))
		retErr := fmt.Errorf("acpx parse events: %w", err)
		emitAgentExited(opts, a.Name(), pid, retErr)
		return resultFromUsage(usage), retErr
	}
	waitErr := started.wait()
	stderrWG.Wait()
	stdinErr := acpxStdinError(<-stdinErrCh)
	if waitErr != nil {
		retErr := fmt.Errorf("acpx exited: %w: %s", errors.Join(waitErr, stdinErr), acpxProcessErrorOutput(stderrBuf, stdoutErr))
		emitAgentExited(opts, a.Name(), pid, retErr)
		return resultFromUsage(usage), retErr
	}
	if stdinErr != nil {
		if out := acpxProcessErrorOutput(stderrBuf, stdoutErr); out != "" {
			stdinErr = fmt.Errorf("%w: %s", stdinErr, out)
		}
		emitAgentExited(opts, a.Name(), pid, stdinErr)
		return resultFromUsage(usage), stdinErr
	}
	res, err := finalizeTextResult(a.Name(), text, opts.JSONSchema, usage)
	if err == nil && res != nil {
		res.SessionID = facts.resumableSessionID(resumeID)
		res.Resumed = resumeID != ""
	}
	emitAgentExited(opts, a.Name(), pid, err)
	return res, err
}

// acpxExecTurn is acpx's one-shot prompt: it opens a fresh ACP session and
// saves no record. Every turn that is not resuming uses it, exactly as every
// acpx turn did before resume existed.
func acpxExecTurn() []string { return []string{"exec", "--file", "-"} }

// acpxPromptTurn is acpx's saved-session prompt, the only form that reconnects
// to an existing ACP session rather than opening a new one.
func acpxPromptTurn(name string) []string {
	return []string{"prompt", "--session", name, "--file", "-"}
}

// acpxSessionName is the acpx session-record name for an ACP session
// identity. acpx keys records by (name, cwd, agent command), so naming the
// record after the identity it carries keeps concurrent runs, worktrees, and
// roles from ever sharing one.
func acpxSessionName(sessionID string) string { return "no-mistakes-" + sessionID }

// loadSession binds an acpx session record to the ACP identity this run
// already minted, which is what makes the prompt that follows reconnect with
// session/load instead of session/new. acpx owns the ACP conversation, so
// `sessions ensure --resume-session` is how a stored identity reaches the
// protocol.
//
// A failure is returned rather than swallowed: the caller in
// pipeline.RunSessions answers a failed resume by dropping the dead identity
// and re-running the same turn cold, so the turn is never skipped.
func (a *acpxAgent) loadSession(ctx context.Context, rawCommand string, opts RunOpts, name, sessionID string) error {
	args := a.buildArgs(rawCommand, opts, []string{"sessions", "ensure", "--name", name, "--resume-session", sessionID})
	cmd := exec.CommandContext(ctx, a.bin, args...)
	cmd.Dir = opts.CWD
	cmd.Env = a.gitSafeEnv(opts.CWD, opts.Env)
	shellenv.ConfigureShellCommand(cmd)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("acpx load session %s: %w: %s", sessionID, err, strings.TrimSpace(string(out)))
	}
	return nil
}

// releaseSession closes the acpx session record a resumed turn used. Only the
// resume path needs it: `exec` is one-shot and leaves nothing behind, while a
// named session is served by a detached acpx queue-owner process that outlives
// the turn's process tree and keeps its agent child alive for the idle TTL. A
// turn cut short leaves that owner mid-prompt, so cancel it first rather than
// letting it keep working on a worktree the run has abandoned.
//
// Closing the record does not end the ACP session, which lives on the target's
// side and stays loadable, so the next round still resumes.
func (a *acpxAgent) releaseSession(ctx context.Context, rawCommand string, opts RunOpts, name string) {
	cleanup, cancel := context.WithTimeout(context.WithoutCancel(ctx), acpxSessionCleanupTimeout)
	defer cancel()
	if ctx.Err() != nil {
		a.runSessionCommand(cleanup, rawCommand, opts, "cancel", "--session", name)
	}
	a.runSessionCommand(cleanup, rawCommand, opts, "sessions", "close", name)
}

// runSessionCommand runs a best-effort acpx session-management command. Its
// outcome never changes the turn's: the prompt has already been answered (or
// abandoned) by the time these run.
func (a *acpxAgent) runSessionCommand(ctx context.Context, rawCommand string, opts RunOpts, command ...string) {
	cmd := exec.CommandContext(ctx, a.bin, a.buildArgs(rawCommand, opts, command)...)
	cmd.Dir = opts.CWD
	cmd.Env = a.gitSafeEnv(opts.CWD, opts.Env)
	shellenv.ConfigureShellCommand(cmd)
	_ = cmd.Run()
}

func (a *acpxAgent) Close() error {
	if a.overlayPath != "" {
		_ = os.Remove(a.overlayPath)
	}
	return nil
}

// resolveRawCommand returns the acpx --agent raw command for this invocation.
// For a neutralized omp gate run it lazily writes the suppression overlay once
// and returns the neutralized `omp acp` command; otherwise it returns the
// configured raw command unchanged.
func (a *acpxAgent) resolveRawCommand() (string, error) {
	if !a.NeutralizesGateInstructions() {
		return a.rawCommand, nil
	}
	a.overlayOnce.Do(func() {
		a.overlayPath, a.overlayErr = writeOMPGateOverlay()
	})
	if a.overlayErr != nil {
		return "", a.overlayErr
	}
	return ompNeutralizedACPCommand(a.overlayPath), nil
}

// buildArgs assembles an acpx invocation: the global options this adapter
// always applies, then the target selection, then command - the acpx
// subcommand and its own arguments (a prompt turn, or one of the session
// commands the resume path uses).
func (a *acpxAgent) buildArgs(rawCommand string, opts RunOpts, command []string) []string {
	args := make([]string, 0, 12+len(command))
	if rawCommand != "" {
		args = append(args, "--agent", rawCommand)
	}
	if opts.CWD != "" {
		args = append(args, "--cwd", opts.CWD)
	}
	args = append(args,
		"--format", "json",
		"--json-strict",
		"--approve-all",
		"--non-interactive-permissions", "deny",
		"--suppress-reads",
	)
	// --model must stay among acpx's own options, ahead of the bare target and
	// the exec subcommand, or acpx reads it as an argument to the target.
	if a.model != "" {
		args = append(args, "--model", a.model)
	}
	if rawCommand == "" {
		args = append(args, a.target)
	}
	return append(args, command...)
}

func acpxStdinError(err error) error {
	if err == nil {
		return nil
	}
	return fmt.Errorf("acpx stdin: %w", err)
}

func acpxProcessErrorOutput(stderr []byte, stdoutErr string) string {
	parts := make([]string, 0, 2)
	if stderrText := strings.TrimSpace(string(stderr)); stderrText != "" {
		parts = append(parts, stderrText)
	}
	if stdoutErr != "" {
		parts = append(parts, stdoutErr)
	}
	return strings.Join(parts, "\n")
}

func buildACPStructuredPrompt(prompt string, schema json.RawMessage) string {
	return prompt + "\n\n## no-mistakes final output contract\n\n" +
		"When the task is complete, your final assistant message must be a single JSON object that matches this JSON Schema. " +
		"Return only the JSON object. Do not wrap it in Markdown fences. Do not include prose before or after the JSON.\n\n" +
		string(schema)
}

// acpxSessionFacts is what one turn's ACP conversation reported about its
// session. acpx relays both directions of the JSON-RPC exchange on stdout, so
// these are the target's own answers rather than anything inferred here.
type acpxSessionFacts struct {
	// loadSession is the agentCapabilities.loadSession flag the target
	// advertised in its `initialize` response.
	loadSession bool
	// sessionID is the identity the target minted in its `session/new`
	// response. A `session/load` answers without one.
	sessionID string
}

// resumableSessionID reports the identity a later turn of the same run may
// resume, or "" when there is none to record. It is empty unless the target
// advertised loadSession: an ACP target that does not advertise it never has
// an identity stored, so nothing ever tries to resume it. A turn that resumed
// re-reports the identity it loaded, because session/load answers without one.
func (f acpxSessionFacts) resumableSessionID(resumeID string) string {
	if !f.loadSession {
		return ""
	}
	if f.sessionID != "" {
		return f.sessionID
	}
	return resumeID
}

type acpxJSONMessage struct {
	Method string         `json:"method"`
	Error  *acpxJSONError `json:"error"`
	Result struct {
		Usage acpxUsageFields `json:"usage"`
		// SessionID is carried by the `session/new` response only. Live
		// session ids elsewhere on the stream ride params, not result.
		SessionID string `json:"sessionId"`
		// AgentCapabilities is carried by the `initialize` response only.
		AgentCapabilities struct {
			LoadSession bool `json:"loadSession"`
		} `json:"agentCapabilities"`
	} `json:"result"`
	Params struct {
		Update acpxSessionUpdate `json:"update"`
	} `json:"params"`
}

type acpxJSONError struct {
	Message string `json:"message"`
}

type acpxSessionUpdate struct {
	SessionUpdate string          `json:"sessionUpdate"`
	Content       json.RawMessage `json:"content"`
	Text          string          `json:"text"`
	Used          int             `json:"used"`
	usedReported  bool
	acpxUsageFields
	Meta struct {
		Usage acpxUsageFields `json:"usage"`
	} `json:"_meta"`
}

type acpxUsageFields struct {
	InputTokens                   int `json:"input_tokens"`
	OutputTokens                  int `json:"output_tokens"`
	CacheReadInputTokens          int `json:"cache_read_input_tokens"`
	CacheReadTokens               int `json:"cache_read_tokens"`
	CacheCreationInputTokens      int `json:"cache_creation_input_tokens"`
	CacheWriteInputTokens         int `json:"cache_write_input_tokens"`
	CacheWriteTokens              int `json:"cache_write_tokens"`
	CachedInputTokens             int `json:"cached_input_tokens"`
	InputTokensCamel              int `json:"inputTokens"`
	OutputTokensCamel             int `json:"outputTokens"`
	CacheReadInputTokensCamel     int `json:"cacheReadInputTokens"`
	CacheCreationInputTokensCamel int `json:"cacheCreationInputTokens"`
	CachedInputTokensCamel        int `json:"cachedInputTokens"`
	CacheReadTokensCamel          int `json:"cacheReadTokens"`
	CachedReadTokensCamel         int `json:"cachedReadTokens"`
	CacheCreationTokensCamel      int `json:"cacheCreationTokens"`
	CacheWriteTokensCamel         int `json:"cacheWriteTokens"`
	CachedWriteTokensCamel        int `json:"cachedWriteTokens"`
	reported                      bool
	cacheCreationReported         bool
}

// parseAcpxJSONEvents streams acpx's JSON events and returns the assistant
// text accumulated so far, on its error paths too, so a turn that fails partway
// can still account for the output acpx already produced.
func parseAcpxJSONEvents(ctx context.Context, r io.Reader, onChunk func(string), usage *TokenUsage, facts *acpxSessionFacts) (string, string, error) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), acpxScannerMaxTokenSize)
	var output strings.Builder
	var stdoutErr string

	for scanner.Scan() {
		select {
		case <-ctx.Done():
			return output.String(), stdoutErr, ctx.Err()
		default:
		}

		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}

		var msg acpxJSONMessage
		if err := json.Unmarshal(line, &msg); err != nil {
			continue
		}
		markAcpxUsagePresence(line, &msg)
		if msg.Error != nil && msg.Error.Message != "" && stdoutErr == "" {
			stdoutErr = msg.Error.Message
		}
		*usage = acpxMaxUsage(*usage, acpxUsageFieldsToTokenUsage(msg.Result.Usage))
		if msg.Result.AgentCapabilities.LoadSession {
			facts.loadSession = true
		}
		if msg.Result.SessionID != "" {
			facts.sessionID = msg.Result.SessionID
		}
		if msg.Method != "session/update" {
			continue
		}

		update := msg.Params.Update
		switch update.SessionUpdate {
		case "usage_update":
			*usage = acpxMaxUsage(*usage, acpxUpdateUsage(update))
		case "agent_message_chunk":
			text := acpxUpdateText(update)
			if text == "" {
				continue
			}
			output.WriteString(text)
			if onChunk != nil {
				onChunk(text)
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return output.String(), stdoutErr, err
	}
	return output.String(), stdoutErr, nil
}

func acpxUpdateUsage(update acpxSessionUpdate) TokenUsage {
	usage := acpxUsageFieldsToTokenUsage(update.acpxUsageFields)
	metaUsage := acpxUsageFieldsToTokenUsage(update.Meta.Usage)
	usage = acpxMaxUsage(usage, metaUsage)
	if update.Used > usage.InputTokens {
		usage.InputTokens = update.Used
	}
	usage.Reported = usage.Reported || update.usedReported || update.Used != 0
	return usage
}

func acpxUsageFieldsToTokenUsage(fields acpxUsageFields) TokenUsage {
	return TokenUsage{
		Reported: fields.reported || acpxUsageFieldsHaveValues(fields),
		CacheCreationReported: fields.cacheCreationReported || acpxFirstPositive(
			fields.CacheCreationInputTokens,
			fields.CacheWriteInputTokens,
			fields.CacheWriteTokens,
			fields.CacheCreationInputTokensCamel,
			fields.CacheCreationTokensCamel,
			fields.CacheWriteTokensCamel,
			fields.CachedWriteTokensCamel,
		) > 0,
		InputTokens: acpxFirstPositive(
			fields.InputTokens,
			fields.InputTokensCamel,
		),
		OutputTokens: acpxFirstPositive(
			fields.OutputTokens,
			fields.OutputTokensCamel,
		),
		CacheReadTokens: acpxFirstPositive(
			fields.CacheReadInputTokens,
			fields.CacheReadTokens,
			fields.CachedInputTokens,
			fields.CacheReadInputTokensCamel,
			fields.CachedInputTokensCamel,
			fields.CacheReadTokensCamel,
			fields.CachedReadTokensCamel,
		),
		CacheCreationTokens: acpxFirstPositive(
			fields.CacheCreationInputTokens,
			fields.CacheWriteInputTokens,
			fields.CacheWriteTokens,
			fields.CacheCreationInputTokensCamel,
			fields.CacheCreationTokensCamel,
			fields.CacheWriteTokensCamel,
			fields.CachedWriteTokensCamel,
		),
	}
}

func markAcpxUsagePresence(line []byte, msg *acpxJSONMessage) {
	var raw struct {
		Result struct {
			Usage json.RawMessage `json:"usage"`
		} `json:"result"`
		Params struct {
			Update json.RawMessage `json:"update"`
		} `json:"params"`
	}
	if json.Unmarshal(line, &raw) != nil {
		return
	}
	markAcpxUsageFields(raw.Result.Usage, &msg.Result.Usage)
	if len(raw.Params.Update) == 0 {
		return
	}
	var update map[string]json.RawMessage
	if json.Unmarshal(raw.Params.Update, &update) != nil {
		return
	}
	markAcpxUsageFields(raw.Params.Update, &msg.Params.Update.acpxUsageFields)
	if _, ok := update["used"]; ok {
		msg.Params.Update.usedReported = true
	}
	if meta, ok := update["_meta"]; ok {
		var metaFields struct {
			Usage json.RawMessage `json:"usage"`
		}
		if json.Unmarshal(meta, &metaFields) == nil {
			markAcpxUsageFields(metaFields.Usage, &msg.Params.Update.Meta.Usage)
		}
	}
}

func markAcpxUsageFields(raw json.RawMessage, fields *acpxUsageFields) {
	if len(raw) == 0 || fields == nil {
		return
	}
	var values map[string]json.RawMessage
	if json.Unmarshal(raw, &values) != nil {
		return
	}
	usageKeys := []string{"input_tokens", "output_tokens", "cache_read_input_tokens", "cache_read_tokens", "cached_input_tokens", "inputTokens", "outputTokens", "cacheReadInputTokens", "cachedInputTokens", "cacheReadTokens", "cachedReadTokens"}
	cacheKeys := []string{"cache_creation_input_tokens", "cache_write_input_tokens", "cache_write_tokens", "cacheCreationInputTokens", "cacheCreationTokens", "cacheWriteTokens", "cachedWriteTokens"}
	for _, key := range usageKeys {
		if _, ok := values[key]; ok {
			fields.reported = true
		}
	}
	for _, key := range cacheKeys {
		if _, ok := values[key]; ok {
			fields.reported = true
			fields.cacheCreationReported = true
		}
	}
}

func acpxUsageFieldsHaveValues(fields acpxUsageFields) bool {
	fields.reported = false
	fields.cacheCreationReported = false
	return fields != (acpxUsageFields{})
}

func acpxMaxUsage(a, b TokenUsage) TokenUsage {
	return TokenUsage{
		Reported:              a.Reported || b.Reported,
		CacheCreationReported: a.CacheCreationReported || b.CacheCreationReported,
		InputTokens:           max(a.InputTokens, b.InputTokens),
		OutputTokens:          max(a.OutputTokens, b.OutputTokens),
		CacheReadTokens:       max(a.CacheReadTokens, b.CacheReadTokens),
		CacheCreationTokens:   max(a.CacheCreationTokens, b.CacheCreationTokens),
	}
}

func acpxFirstPositive(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

func acpxUpdateText(update acpxSessionUpdate) string {
	if update.Text != "" {
		return update.Text
	}
	if len(update.Content) == 0 {
		return ""
	}
	var content struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(update.Content, &content); err == nil && content.Text != "" {
		return content.Text
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(update.Content, &parts); err != nil {
		return ""
	}
	var b strings.Builder
	for _, part := range parts {
		if part.Text != "" {
			b.WriteString(part.Text)
		}
	}
	return b.String()
}

func estimateAcpxTokens(charCount int) int {
	if charCount <= 0 {
		return 0
	}
	return (charCount + 3) / 4
}
