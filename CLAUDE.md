# NanoClaw

Personal assistant powered by Ollama (local LLM). See [README.md](README.md) for philosophy and setup. See [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) for architecture decisions.

## Quick Context

Single Node.js process with skill-based channel system. Channels (WhatsApp, Telegram, Slack, Discord, Gmail) are skills that self-register at startup. Messages route to an Ollama agent engine (Vercel AI SDK) running on the host with GPU access. Bash/file tools execute via `docker exec` into a minimal per-group Ubuntu sandbox container for isolation. Each group has isolated filesystem and memory.

**Engine:** Ollama (`mistral-small3.2:24b` selected after benchmark, configurable via `OLLAMA_MODEL` in `.env`)
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
| `webSearch` | DuckDuckGo search (news, facts, prices, dates) |
| `webFetch` | HTTP fetch (text or JSON) |
| `gmailSearch` | Search user's Gmail inbox (main group only) |
| `gmailRead` | Read a specific email by ID (main group only) |
| `gmailSend` | Send email (main group only) |
| `sendMessage` | Send intermediate message to user during work |
| `scheduleTask` | Create cron/interval/once scheduled tasks |
| `listTasks` | List scheduled tasks |
| `deleteTask` | Delete a scheduled task by ID |
| `cancelAllTasks` | Delete all scheduled tasks |
| `remember` | Save a fact to persistent memory |
| `setModel` | Change Ollama model at runtime |

## System Prompt & Tool Routing

Le system prompt de l'agent est construit en couches dans `buildSystemPrompt()` (`src/agent-engine.ts` L212+) :

1. `groups/global/CLAUDE.md` — instructions communes (langue, ton, formatting, météo)
2. `groups/{name}/CLAUDE.md` — instructions par groupe (override)
3. Section hardcodée "CRITICAL: Tool Usage" — routes tool et règles anti-boucle

**Routes tool** (section hardcodée, L233+) : quand un nouveau tool est ajouté, il faut aussi ajouter sa route dans cette section pour que le LLM sache quand l'utiliser. Format : `• User intent → toolName(...)`.

**Règles comportementales** (même section) :
- Agir immédiatement si la requête est claire, sinon demander clarification
- Anti-boucle : max 3 tool calls sans résultat → stop et demande à l'utilisateur
- Jamais le même type de tool 2+ fois de suite

## Configuration (.env)

```
ASSISTANT_NAME=Oliv
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral-small3.2:24b
AGENT_MAX_STEPS=15
TELEGRAM_BOT_TOKEN=...
```

## Troubleshooting

**Agent ne répond pas :** vérifier `logs/nanoclaw.log` — souvent un modèle Ollama non installé (`curl http://localhost:11434/api/tags`).

**WhatsApp not connecting after upgrade:** WhatsApp is now a separate channel fork, not bundled in core. Run `/add-whatsapp` to install it. Existing auth credentials and groups are preserved.

## Container Build Cache

The container buildkit caches the build context aggressively. `--no-cache` alone does NOT invalidate COPY steps — the builder's volume retains stale files. To force a truly clean rebuild, prune the builder then re-run `./container/build.sh`.
