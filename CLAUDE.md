# NanoClaw

Personal assistant powered by Ollama (local LLM). See [README.md](README.md) for philosophy and setup. See [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) for architecture decisions.

## Quick Context

Single Node.js process with skill-based channel system. Channels (WhatsApp, Telegram, Slack, Discord, Gmail) are skills that self-register at startup. Messages route to an Ollama agent engine (Vercel AI SDK) running on the host with GPU access. Bash/file tools execute via `docker exec` into a minimal per-group Ubuntu sandbox container for isolation. Each group has isolated filesystem and memory.

**Engine:** Ollama (`qwen3.5:cloud` by default, configurable via `OLLAMA_MODEL` in `.env`)
**Sandbox:** `nanoclaw-agent:latest` — Ubuntu 24.04 minimal, `sleep infinity`, no Node.js/Chromium

## Key Files

| File | Purpose |
|------|---------|
| `src/index.ts` | Orchestrator: state, message loop, agent invocation |
| `src/agent-engine.ts` | Ollama agentic loop (Vercel AI SDK), tool definitions, sandbox lifecycle |
| `src/container-runner.ts` | Volume mounts, delegates to agent-engine |
| `src/channels/registry.ts` | Channel registry (self-registration at startup) |
| `src/router.ts` | Message formatting and outbound routing |
| `src/config.ts` | Trigger pattern, paths, intervals, Ollama config |
| `src/task-scheduler.ts` | Runs scheduled tasks |
| `src/db.ts` | SQLite operations |
| `groups/{name}/CLAUDE.md` | Per-group system prompt (agent reads at startup) |
| `container/Dockerfile` | Ubuntu 24.04 minimal sandbox (bash, curl, git, python3, jq) |

## Skills

| Skill | When to Use |
|-------|-------------|
| `/setup` | First-time installation, authentication, service configuration |
| `/customize` | Adding channels, integrations, changing behavior |
| `/debug` | Container issues, logs, troubleshooting |
| `/update-nanoclaw` | Bring upstream NanoClaw updates into a customized install |
| `/qodo-pr-resolver` | Fetch and fix Qodo PR review issues interactively or in batch |
| `/get-qodo-rules` | Load org- and repo-level coding rules from Qodo before code tasks |

## Development

Run commands directly—don't tell the user to run them.

```bash
npm run dev          # Run with hot reload
npm run build        # Compile TypeScript
./container/build.sh # Rebuild agent container
```

Service management:
```bash
# macOS (launchd)
launchctl load ~/Library/LaunchAgents/com.nanoclaw.plist
launchctl unload ~/Library/LaunchAgents/com.nanoclaw.plist
launchctl kickstart -k gui/$(id -u)/com.nanoclaw  # restart

# Linux (systemd)
systemctl --user start nanoclaw
systemctl --user stop nanoclaw
systemctl --user restart nanoclaw
```

## Agent Tools (available in sandbox)

| Tool | Description |
|------|-------------|
| `bash` | Execute bash commands via `docker exec` in the sandbox |
| `readFile` | Read files from `/workspace/group` or `/workspace/global` |
| `writeFile` | Write files to `/workspace/group` |
| `webFetch` | HTTP fetch (text or JSON) |
| `sendMessage` | Send intermediate message to user during work |
| `scheduleTask` | Create cron/interval/once scheduled tasks |

## Configuration (.env)

```
ASSISTANT_NAME=Oliv
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3.5:cloud
AGENT_MAX_STEPS=50
TELEGRAM_BOT_TOKEN=...
```

## Troubleshooting

**Agent ne répond pas :** vérifier `logs/nanoclaw.log` — souvent un modèle Ollama non installé (`curl http://localhost:11434/api/tags`).

**WhatsApp not connecting after upgrade:** WhatsApp is now a separate channel fork, not bundled in core. Run `/add-whatsapp` to install it. Existing auth credentials and groups are preserved.

## Container Build Cache

The container buildkit caches the build context aggressively. `--no-cache` alone does NOT invalidate COPY steps — the builder's volume retains stale files. To force a truly clean rebuild, prune the builder then re-run `./container/build.sh`.
