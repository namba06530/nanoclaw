/**
 * Agent Engine for NanoClaw — Ollama-powered (Vercel AI SDK v6)
 *
 * Replaces the Anthropic Claude Code SDK. The LLM (Ollama) runs on the host
 * with GPU access. Bash/file tools execute via `docker exec` into a minimal
 * per-group sandbox container for isolation.
 */
import { exec as execCallback, spawn } from 'child_process';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { promisify } from 'util';

import { google } from 'googleapis';

import { generateText, stepCountIs, tool } from 'ai';
import { createOllama } from 'ollama-ai-provider-v2';
import { z } from 'zod';

import {
  ContainerInput,
  ContainerOutput,
  VolumeMount,
} from './container-runner.js';
import {
  AGENT_MAX_STEPS,
  CONVERSATION_HISTORY_TURNS,
  GROUPS_DIR,
  OLLAMA_BASE_URL,
  OLLAMA_MODEL,
  TIMEZONE,
} from './config.js';
import {
  clearConversationHistory,
  getConversationHistory,
  storeConversationTurn,
} from './db.js';
import { resolveGroupFolderPath } from './group-folder.js';
import { logger } from './logger.js';
import { RegisteredGroup, ScheduledTask } from './types.js';
import { readEnvFile, writeEnvFile } from './env.js';

const execAsync = promisify(execCallback);

// Jina API key — optional, improves search quality and rate limits
const { JINA_API_KEY } = readEnvFile(['JINA_API_KEY']);
const jinaApiKey = process.env.JINA_API_KEY || JINA_API_KEY || '';

// Mutable model — can be changed at runtime via setModel tool
let currentModel = OLLAMA_MODEL;

const MAX_STEPS = AGENT_MAX_STEPS;
const BASH_TIMEOUT_MS = 30_000;

// ─── Gmail helper ─────────────────────────────────────────────────────────────

// Convert French dates in a query to Gmail date operators
// "digitalocean 13 mars" → "digitalocean after:2026/03/12 before:2026/03/14"
// "edf 17 mars 2026" → "edf after:2026/03/16 before:2026/03/18"
const FRENCH_MONTHS: Record<string, number> = {
  janvier: 1,
  février: 2,
  fevrier: 2,
  mars: 3,
  avril: 4,
  mai: 5,
  juin: 6,
  juillet: 7,
  août: 8,
  aout: 8,
  septembre: 9,
  octobre: 10,
  novembre: 11,
  décembre: 12,
  decembre: 12,
};

function preprocessGmailQuery(query: string): string {
  const monthPattern = Object.keys(FRENCH_MONTHS).join('|');
  const dateRegex = new RegExp(
    `(\\d{1,2})\\s+(${monthPattern})(?:\\s+(\\d{4}))?`,
    'i',
  );
  const match = query.match(dateRegex);
  if (!match) return query;

  const day = parseInt(match[1]);
  const month = FRENCH_MONTHS[match[2].toLowerCase()];
  const year = match[3] ? parseInt(match[3]) : new Date().getFullYear();

  const target = new Date(year, month - 1, day);
  const before = new Date(target);
  before.setDate(before.getDate() + 1);

  const fmt = (d: Date) =>
    `${d.getFullYear()}/${String(d.getMonth() + 1).padStart(2, '0')}/${String(d.getDate()).padStart(2, '0')}`;
  const cleaned = query.replace(match[0], '').trim();
  return `${cleaned} after:${fmt(target)} before:${fmt(before)}`;
}

// Singleton — one client per process to avoid token refresh conflicts
let _gmailClient: ReturnType<typeof google.gmail> | null = null;

function createGmailClient() {
  if (_gmailClient) return _gmailClient;

  const credDir = path.join(os.homedir(), '.gmail-mcp');
  const keysPath = path.join(credDir, 'gcp-oauth.keys.json');
  const tokensPath = path.join(credDir, 'credentials.json');

  if (!fs.existsSync(keysPath) || !fs.existsSync(tokensPath)) return null;

  const keys = JSON.parse(fs.readFileSync(keysPath, 'utf-8'));
  const tokens = JSON.parse(fs.readFileSync(tokensPath, 'utf-8'));
  const cfg = keys.installed || keys.web || keys;

  const auth = new google.auth.OAuth2(
    cfg.client_id,
    cfg.client_secret,
    cfg.redirect_uris?.[0],
  );
  auth.setCredentials(tokens);

  // Persist refreshed tokens automatically
  auth.on('tokens', (newTokens) => {
    try {
      const current = JSON.parse(fs.readFileSync(tokensPath, 'utf-8'));
      Object.assign(current, newTokens);
      fs.writeFileSync(tokensPath, JSON.stringify(current, null, 2));
    } catch {}
  });

  _gmailClient = google.gmail({ version: 'v1', auth });
  return _gmailClient;
}

// Persistent sandbox container per group (started once, reused across messages)
const sandboxContainers = new Map<string, string>();

export interface ScheduleParams {
  prompt: string;
  schedule_type: 'cron' | 'interval' | 'once';
  schedule_value: string;
  context_mode?: 'group' | 'isolated' | 'watch';
  target_group_jid?: string;
}

export interface AgentCallbacks {
  sendMessage: (jid: string, text: string) => Promise<void>;
  scheduleTask: (
    groupFolder: string,
    isMain: boolean,
    chatJid: string,
    params: ScheduleParams,
  ) => Promise<void>;
  getAvailableGroups: () => Array<{
    jid: string;
    name: string;
    lastActivity: string;
    isRegistered: boolean;
  }>;
  getAllTasks: () => ScheduledTask[];
  deleteTask: (id: string) => void;
  updateTask: (
    id: string,
    updates: Partial<
      Pick<
        ScheduledTask,
        'prompt' | 'schedule_type' | 'schedule_value' | 'status'
      >
    >,
  ) => void;
  registerGroup: (
    jid: string,
    name: string,
    folder: string,
    trigger: string,
  ) => void;
}

// ─── Sandbox lifecycle ────────────────────────────────────────────────────────

/**
 * Ensure a sandbox container is running for the group.
 * Starts a new container if needed, reuses existing one.
 */
export async function ensureSandbox(
  group: RegisteredGroup,
  mounts: VolumeMount[],
): Promise<string> {
  const safeName = group.folder.replace(/[^a-zA-Z0-9-]/g, '-');
  const containerName = `nanoclaw-sandbox-${safeName}`;

  // Check if our known container is still alive
  const existing = sandboxContainers.get(group.folder);
  if (existing === containerName) {
    try {
      const { stdout } = await execAsync(
        `docker inspect --format='{{.State.Running}}' ${containerName}`,
      );
      if (stdout.trim() === 'true') {
        return containerName;
      }
    } catch {
      // Container gone
    }
    sandboxContainers.delete(group.folder);
  }

  // Remove stale container with the same name if it exists
  await execAsync(`docker rm -f ${containerName}`).catch(() => {});

  // Build mount args
  const mountParts = mounts.map((m) =>
    m.readonly
      ? `--mount type=bind,src=${m.hostPath},dst=${m.containerPath},readonly`
      : `-v ${m.hostPath}:${m.containerPath}`,
  );

  await execAsync(
    `docker run -d --name ${containerName} --rm -e TZ=${TIMEZONE} ${mountParts.join(' ')} nanoclaw-agent:latest`,
  );

  sandboxContainers.set(group.folder, containerName);
  logger.info(
    { group: group.name, containerName },
    'Sandbox container started',
  );
  return containerName;
}

/**
 * Stop a group's sandbox container (called on shutdown).
 */
export async function stopSandbox(groupFolder: string): Promise<void> {
  const name = sandboxContainers.get(groupFolder);
  if (!name) return;
  sandboxContainers.delete(groupFolder);
  await execAsync(`docker stop ${name}`).catch(() => {});
  logger.info(
    { groupFolder, containerName: name },
    'Sandbox container stopped',
  );
}

/**
 * Stop all running sandbox containers (called on nanoclaw shutdown).
 */
export async function stopAllSandboxes(): Promise<void> {
  const folders = Array.from(sandboxContainers.keys());
  await Promise.all(folders.map(stopSandbox));
}

// ─── System prompt ────────────────────────────────────────────────────────────

function buildSystemPrompt(
  group: RegisteredGroup,
  isMain: boolean,
  callbacks: AgentCallbacks,
  isScheduledTask = false,
): string {
  const parts: string[] = [];

  // Global CLAUDE.md (for all agents)
  const globalMd = path.join(GROUPS_DIR, 'global', 'CLAUDE.md');
  if (fs.existsSync(globalMd)) {
    parts.push(fs.readFileSync(globalMd, 'utf-8'));
  }

  // Per-group CLAUDE.md (non-main groups, or main group's own CLAUDE.md)
  const groupMd = path.join(resolveGroupFolderPath(group.folder), 'CLAUDE.md');
  if (fs.existsSync(groupMd)) {
    parts.push('\n\n---\n\n' + fs.readFileSync(groupMd, 'utf-8'));
  }

  // CRITICAL: Tool usage instructions — placed early for model attention
  parts.push(
    '\n\n## CRITICAL: Tool Usage\n' +
      'You MUST call a tool for every state-changing action. The action does NOT happen until the tool returns a result.\n' +
      'NEVER fabricate factual data — if you did not search, you do not know.\n' +
      'NEVER confirm success BEFORE receiving the tool result.\n\n' +
      'Required tool routing — always use the most specific tool:\n' +
      '• Weather → webFetch("https://wttr.in/{city}?format=j1")\n' +
      '• Emails/mails → gmailSearch(keyword). Body preview included — summarize directly.\n' +
      '• Delete email → gmailDelete(keyword). Single match = trashed. Multiple matches = refine query.\n' +
      '• Tasks → listTasks()\n' +
      '• Schedule → scheduleTask(...) → wait for result → confirm\n' +
      '• Delete tasks → deleteTask(id) or cancelAllTasks() → wait → confirm\n' +
      '• Send email → gmailSend(...) → wait → confirm\n' +
      '• Save/remember → remember(...) → wait → confirm\n' +
      '• News, scores, dates, prices, facts → webSearch([query]) → report\n\n' +
      'Act immediately — do NOT ask "do you want me to check?" Pick the most specific tool and proceed.\n' +
      'Ask the user to clarify ONLY when you genuinely cannot determine what they want.\n\n' +
      'ANTI-LOOP: If after 3 tool calls you still do not have a good answer, STOP.\n' +
      'Tell the user what you tried and ask how to proceed.\n' +
      'Never call the same tool type (e.g. webFetch) more than 2 times in a row.\n' +
      'If a tool returns 0 results or an error, ALWAYS tell the user explicitly what you searched and that nothing was found. Never send an empty response.',
  );

  // Persistent memory (memory.json in group folder)
  const memoryFile = path.join(
    resolveGroupFolderPath(group.folder),
    'memory.json',
  );
  if (fs.existsSync(memoryFile)) {
    try {
      const memory = JSON.parse(fs.readFileSync(memoryFile, 'utf-8'));
      if (Object.keys(memory).length > 0) {
        parts.push(
          '\n\n## Persistent Memory\n' + JSON.stringify(memory, null, 2),
        );
      }
    } catch {
      /* corrupted file — ignore */
    }
  }

  // Scheduled tasks — tool-only access (no JSON injection)
  parts.push(
    '\n\n## Scheduled Tasks\n' +
      'Use the listTasks tool to view current tasks. ' +
      'Use deleteTask, cancelAllTasks, updateTask, or scheduleTask to modify them.',
  );

  // Available groups (main only)
  if (isMain) {
    const groups = callbacks.getAvailableGroups();
    parts.push('\n\n## Available Groups\n' + JSON.stringify(groups, null, 2));
  }

  // Scheduled task context
  if (isScheduledTask) {
    parts.push(
      '\n\n## Scheduled Task Execution\n' +
        'You are running as a scheduled task. ' +
        'Your text output is automatically sent to the user — UNLESS this is a watch task.\n' +
        'For watch tasks: follow the task prompt exactly. ' +
        'Use the sendMessage tool ONLY if the monitored condition is met. ' +
        'Return nothing (no text output) if the condition is not triggered.',
    );
  }

  // Memory instructions, response style, attached files
  parts.push(
    '\n\n## Proactive Memory\n' +
      'You have a `remember` tool to save information across conversations. ' +
      'Use it proactively whenever you learn something useful about a user: ' +
      'their city, preferences, favorite teams, habits, recurring requests, etc. ' +
      'Use descriptive keys, e.g. "fab_city", "olivier_football_team", "preferred_language". ' +
      "Do NOT save ephemeral info (today's weather, one-off sports results).\n\n" +
      '## Response Instructions\n' +
      'Be concise. Use the language of the user. ' +
      'Do not add unnecessary disclaimers or repetition. ' +
      'For messaging apps: keep responses short unless detail is explicitly requested.\n\n' +
      '## Attached files\n' +
      'When a message starts with `Fichier joint "nom" :`, the file content has already been extracted and is provided directly in the message. ' +
      'Read and use it directly — no tool needed. ' +
      'If the content says "[PDF scanné ...]", inform the user that OCR is unavailable for that scanned document.',
  );

  return parts.join('\n');
}

// ─── Tool definitions ─────────────────────────────────────────────────────────

function buildTools(
  containerName: string,
  group: RegisteredGroup,
  input: ContainerInput,
  callbacks: AgentCallbacks,
) {
  return {
    bash: tool({
      description:
        'Execute a bash command in the isolated sandbox container. Working directory is /workspace/group.',
      inputSchema: z.object({
        command: z.string().describe('The bash command to execute'),
      }),
      execute: async ({ command }) => {
        try {
          const { stdout, stderr } = await execAsync(
            `docker exec -w /workspace/group ${containerName} bash -c ${JSON.stringify(command)}`,
            { timeout: BASH_TIMEOUT_MS },
          );
          return { stdout: stdout || '', stderr: stderr || '', exitCode: 0 };
        } catch (err: any) {
          return {
            stdout: err.stdout || '',
            stderr: err.stderr || err.message,
            exitCode: err.code ?? 1,
          };
        }
      },
    }),

    readFile: tool({
      description:
        'Read a file from the sandbox filesystem (/workspace/group or /workspace/global)',
      inputSchema: z.object({
        path: z
          .string()
          .describe(
            'Absolute path in the container (e.g. /workspace/group/notes.md)',
          ),
      }),
      execute: async ({ path: filePath }) => {
        if (!filePath.startsWith('/workspace/')) {
          return { error: 'Access denied: path must start with /workspace/' };
        }
        try {
          const { stdout } = await execAsync(
            `docker exec ${containerName} cat ${JSON.stringify(filePath)}`,
          );
          return { content: stdout };
        } catch (err: any) {
          return { error: err.message };
        }
      },
    }),

    writeFile: tool({
      description:
        'Write content to a file in /workspace/group (persistent, visible in groups/{folder}/)',
      inputSchema: z.object({
        path: z
          .string()
          .describe(
            'Absolute path in the container, must be under /workspace/group/',
          ),
        content: z.string().describe('Content to write'),
      }),
      execute: async ({ path: filePath, content }) => {
        if (!filePath.startsWith('/workspace/group/')) {
          return {
            error: 'Access denied: can only write to /workspace/group/',
          };
        }
        return new Promise<{ success: boolean } | { error: string }>(
          (resolve) => {
            const proc = spawn('docker', [
              'exec',
              '-i',
              containerName,
              'bash',
              '-c',
              `mkdir -p "$(dirname ${JSON.stringify(filePath)})" && cat > ${JSON.stringify(filePath)}`,
            ]);
            proc.stdin.write(content);
            proc.stdin.end();
            proc.on('close', (code) =>
              resolve(
                code === 0
                  ? { success: true }
                  : { error: `Write exited with code ${code}` },
              ),
            );
            proc.on('error', (err) => resolve({ error: err.message }));
          },
        );
      },
    }),

    webFetch: tool({
      description:
        'Fetch content from a URL. For web pages, returns clean readable text (HTML → markdown). For JSON APIs, use format: "json".',
      inputSchema: z.object({
        url: z.string().describe('URL to fetch'),
        format: z
          .enum(['text', 'json'])
          .default('text')
          .describe(
            'Use "text" for web pages (returns clean markdown), "json" for JSON APIs (direct fetch)',
          ),
      }),
      execute: async ({ url, format }) => {
        try {
          if (format === 'json') {
            const res = await fetch(url, {
              headers: { 'User-Agent': 'NanoClaw/1.0' },
              signal: AbortSignal.timeout(15_000),
            });
            const text = await res.text();
            try {
              return { content: JSON.parse(text), status: res.status };
            } catch {
              return { content: text.slice(0, 50_000), status: res.status };
            }
          }
          // Proxy through Jina Reader: converts any URL to clean markdown
          const jinaUrl = `https://r.jina.ai/${url}`;
          const jinaHeaders: Record<string, string> = {
            'User-Agent': 'NanoClaw/1.0',
            'X-Retain-Images': 'none',
          };
          if (jinaApiKey) jinaHeaders['Authorization'] = `Bearer ${jinaApiKey}`;
          const res = await fetch(jinaUrl, {
            headers: jinaHeaders,
            signal: AbortSignal.timeout(20_000),
          });
          const text = await res.text();
          return { content: text.slice(0, 30_000), status: res.status };
        } catch (err: any) {
          return { error: err.message };
        }
      },
    }),

    sendMessage: tool({
      description:
        'Send a message to the user immediately while still working on a task',
      inputSchema: z.object({
        text: z.string().describe('The message to send'),
      }),
      execute: async ({ text }) => {
        try {
          await callbacks.sendMessage(input.chatJid, text);
          return { success: true };
        } catch (err: any) {
          return { error: err.message };
        }
      },
    }),

    sendMessageTo: tool({
      description:
        'Send a Telegram message to a named contact from contacts.json. ' +
        'This tool handles contact resolution internally — do NOT use readFile to look up contacts first. ' +
        'IMPORTANT: Before calling this tool, always use sendMessage to show the user ' +
        'the draft (contact name + message text) and ask for explicit confirmation. ' +
        'Only call sendMessageTo after the user has confirmed.',
      inputSchema: z.object({
        contact_name: z
          .string()
          .describe('Name of the contact as defined in contacts.json'),
        message: z.string().describe('Message text to send'),
      }),
      execute: async ({ contact_name, message }) => {
        try {
          const contactsPath = path.join(
            GROUPS_DIR,
            group.folder,
            'contacts.json',
          );
          if (!fs.existsSync(contactsPath)) {
            return {
              error:
                'contacts.json not found in group folder. Create it with a "contacts" object mapping names to tg:<chatId> JIDs.',
            };
          }
          const raw = JSON.parse(fs.readFileSync(contactsPath, 'utf-8'));
          const contacts: Record<string, string> = raw.contacts || {};

          const key = Object.keys(contacts).find(
            (k) => k.toLowerCase() === contact_name.toLowerCase(),
          );
          if (!key) {
            const available = Object.keys(contacts).join(', ') || '(none)';
            return {
              error: `Contact "${contact_name}" not found. Available: ${available}`,
            };
          }

          const jid = contacts[key];
          if (!jid.startsWith('tg:')) {
            return {
              error: `JID "${jid}" is not a Telegram JID (must start with tg:). Only Telegram is supported.`,
            };
          }

          await callbacks.sendMessage(jid, message);
          return { success: true, sent_to: key, jid };
        } catch (err: any) {
          return { error: err.message };
        }
      },
    }),

    ...(input.isMain
      ? {
          registerGroup: tool({
            description:
              'Register a new chat (Telegram or WhatsApp) so the agent can respond to messages from it. ' +
              'Use this when the user wants to add a new group or private chat. ' +
              'The folder name must be unique, lowercase, alphanumeric with hyphens/underscores only.',
            inputSchema: z.object({
              jid: z
                .string()
                .describe(
                  'Chat JID — Telegram: "tg:<chatId>", WhatsApp: "<number>@s.whatsapp.net"',
                ),
              name: z
                .string()
                .describe('Display name for this group (e.g. "Le Clan")'),
              folder: z
                .string()
                .describe(
                  'Folder name for storage — lowercase, alphanumeric, hyphens/underscores (e.g. "le-clan")',
                ),
              trigger: z
                .string()
                .default('@Oliv')
                .describe(
                  'Trigger pattern to activate the agent (e.g. "@Oliv")',
                ),
            }),
            execute: async ({ jid, name, folder, trigger }) => {
              try {
                callbacks.registerGroup(jid, name, folder, trigger);
                return { success: true, jid, name, folder, trigger };
              } catch (err: any) {
                return { error: err.message };
              }
            },
          }),
        }
      : {}),

    scheduleTask: tool({
      description:
        'Schedule a task to run automatically (cron, interval, or once)',
      inputSchema: z.object({
        prompt: z
          .string()
          .describe('Instructions for the agent when the task runs'),
        schedule_type: z
          .enum(['cron', 'interval', 'once'])
          .describe(
            'cron: cron expression, interval: ms between runs, once: ISO datetime',
          ),
        schedule_value: z
          .string()
          .describe(
            'Value: cron expr (e.g. "0 9 * * *"), ms (e.g. "3600000"), or ISO datetime',
          ),
        context_mode: z
          .enum(['group', 'isolated', 'watch'])
          .default('group')
          .describe(
            'group: auto-sends output, uses conversation context. ' +
              'isolated: auto-sends output, fresh session. ' +
              'watch: suppresses auto-send — agent uses sendMessage tool only when condition is met.',
          ),
        target_group_jid: z
          .string()
          .optional()
          .describe(
            'Target group JID (main group only, to schedule for other groups)',
          ),
      }),
      execute: async (params) => {
        try {
          await callbacks.scheduleTask(
            group.folder,
            input.isMain,
            input.chatJid,
            params,
          );
          return { success: true };
        } catch (err: any) {
          return { error: err.message };
        }
      },
    }),

    listTasks: tool({
      description:
        'List all scheduled tasks. Call this when the user asks about their tasks.',
      inputSchema: z.object({}),
      execute: async () => {
        const allTasks = callbacks.getAllTasks();
        const tasks = input.isMain
          ? allTasks
          : allTasks.filter((t) => t.group_folder === group.folder);
        if (tasks.length === 0) {
          return { tasks: [], message: 'No scheduled tasks' };
        }
        return {
          tasks: tasks.map((t) => ({
            id: t.id,
            prompt: t.prompt.slice(0, 100),
            schedule: `${t.schedule_type}: ${t.schedule_value}`,
            status: t.status,
            context_mode: t.context_mode,
            next_run: t.next_run,
          })),
        };
      },
    }),

    deleteTask: tool({
      description:
        'Permanently delete a scheduled task by its ID. Use when the user wants to cancel or remove a specific task.',
      inputSchema: z.object({
        task_id: z.string().describe('The task ID to delete'),
      }),
      execute: async ({ task_id }) => {
        try {
          const all = callbacks.getAllTasks();
          const task = all.find((t) => t.id === task_id);
          if (!task) return { error: `Task "${task_id}" not found` };
          if (!input.isMain && task.group_folder !== group.folder) {
            return {
              error: 'Unauthorized: can only delete tasks from your own group',
            };
          }
          callbacks.deleteTask(task_id);
          return { success: true, deleted: task_id };
        } catch (err: any) {
          return { error: err.message };
        }
      },
    }),

    cancelAllTasks: tool({
      description:
        'Cancel and delete ALL scheduled tasks in a single call. Use this when the user wants to stop or remove all tasks — do NOT loop deleteTask manually.',
      inputSchema: z.object({}),
      execute: async () => {
        try {
          // Main group: delete all visible tasks; non-main: own group only
          const allTasks = callbacks.getAllTasks();
          const targets = input.isMain
            ? allTasks
            : allTasks.filter((t) => t.group_folder === group.folder);
          for (const t of targets) {
            callbacks.deleteTask(t.id);
          }
          return { success: true, cancelled: targets.length };
        } catch (err: any) {
          return { error: err.message };
        }
      },
    }),

    updateTask: tool({
      description:
        'Modify a scheduled task: pause it, resume it, change its schedule, or update its prompt.',
      inputSchema: z.object({
        task_id: z.string().describe('The task ID to update'),
        status: z
          .enum(['active', 'paused'])
          .optional()
          .describe('Set to "paused" to pause, "active" to resume'),
        schedule_type: z
          .enum(['cron', 'interval', 'once'])
          .optional()
          .describe('New schedule type'),
        schedule_value: z
          .string()
          .optional()
          .describe('New schedule value (cron expr, ms, or ISO datetime)'),
        prompt: z
          .string()
          .optional()
          .describe('New prompt/instructions for the task'),
      }),
      execute: async ({
        task_id,
        status,
        schedule_type,
        schedule_value,
        prompt,
      }) => {
        try {
          const all = callbacks.getAllTasks();
          const task = all.find((t) => t.id === task_id);
          if (!task) return { error: `Task "${task_id}" not found` };
          if (!input.isMain && task.group_folder !== group.folder) {
            return {
              error: 'Unauthorized: can only update tasks from your own group',
            };
          }
          callbacks.updateTask(task_id, {
            status,
            schedule_type,
            schedule_value,
            prompt,
          });
          return { success: true, updated: task_id };
        } catch (err: any) {
          return { error: err.message };
        }
      },
    }),

    clearHistory: tool({
      description:
        'Clear the conversation history for this group — use when the user asks to forget, reset, or start fresh',
      inputSchema: z.object({}),
      execute: async () => {
        clearConversationHistory(group.folder);
        return { success: true, message: 'Conversation history cleared' };
      },
    }),

    remember: tool({
      description:
        'Save a piece of information to persistent memory (survives across conversations and restarts). Use an empty string as value to delete a key.',
      inputSchema: z.object({
        key: z
          .string()
          .describe('Memory key, e.g. "user_city", "preferred_language"'),
        value: z
          .string()
          .describe('Value to store. Pass empty string to delete the key.'),
      }),
      execute: async ({ key, value }) => {
        const memFile = path.join(
          resolveGroupFolderPath(group.folder),
          'memory.json',
        );
        let memory: Record<string, string> = {};
        try {
          memory = JSON.parse(fs.readFileSync(memFile, 'utf-8'));
        } catch {}
        if (value === '') {
          delete memory[key];
        } else {
          memory[key] = value;
        }
        fs.writeFileSync(memFile, JSON.stringify(memory, null, 2));
        return { success: true };
      },
    }),

    webSearch: tool({
      description:
        'Search the web and return structured results (title, url, snippet). Use for current events, facts, or when you need to find specific information online.',
      inputSchema: z.object({
        query: z.string().describe('Search query'),
      }),
      execute: async ({ query }) => {
        const encoded = encodeURIComponent(query);

        // Helper: extract snippet from Jina item
        const extractSnippet = (item: any): string => {
          let snippet = (item.description || '').trim();
          if (!snippet && item.content) {
            for (const line of (item.content as string).split('\n')) {
              const clean = line.replace(/^[#*\-\[\]>|]+\s*/g, '').trim();
              if (clean.length > 60 && !clean.startsWith('http')) {
                snippet = clean;
                break;
              }
            }
          }
          return snippet.slice(0, 300);
        };

        // Helper: DDG fallback via Jina Reader
        // DDG renders each result as 2-3 links to the same URL:
        //   1. short title ("Peymeinade - Wikipedia")
        //   2. snippet text ("Peymeinade is a commune in the Alpes-Maritimes...")
        //   3. bare domain ("en.wikipedia.org/wiki/Peymeinade")
        // Strategy: deduplicate by URL, keep the LONGEST title as the snippet.
        const searchDDG = async () => {
          const jinaUrl = `https://r.jina.ai/https://html.duckduckgo.com/html/?q=${encoded}`;
          const res = await fetch(jinaUrl, {
            headers: {
              'User-Agent': 'NanoClaw/1.0',
              'X-Retain-Images': 'none',
            },
            signal: AbortSignal.timeout(15_000),
          });
          const markdown = await res.text();

          // Collect all matches, keeping longest title per URL
          const byUrl = new Map<string, { title: string; snippet: string }>();
          const linkPattern =
            /\[([^\]]{5,})\]\(https:\/\/duckduckgo\.com\/l\/\?uddg=([^&)]+)[^)]*\)/g;
          let match;
          while ((match = linkPattern.exec(markdown)) !== null) {
            const text = match[1].trim();
            const realUrl = decodeURIComponent(match[2]);
            if (
              !realUrl.startsWith('http') ||
              realUrl.includes('duckduckgo.com')
            )
              continue;
            const existing = byUrl.get(realUrl);
            if (!existing) {
              byUrl.set(realUrl, { title: text, snippet: '' });
            } else if (text.length > existing.title.length) {
              // Longer text is the snippet; keep shorter as title
              byUrl.set(realUrl, {
                title: existing.title,
                snippet: text.slice(0, 300),
              });
            }
          }

          return Array.from(byUrl.entries())
            .slice(0, 5)
            .map(([url, { title, snippet }]) => ({ title, url, snippet }));
        };

        try {
          if (jinaApiKey) {
            // Race s.jina.ai (rich results) against DDG fallback (reliable ~3s)
            // Whichever resolves first with non-empty results wins
            const jinaSearch = fetch(`https://s.jina.ai/${encoded}`, {
              headers: {
                'User-Agent': 'NanoClaw/1.0',
                Accept: 'application/json',
                'X-Retain-Images': 'none',
                Authorization: `Bearer ${jinaApiKey}`,
              },
              signal: AbortSignal.timeout(25_000),
            })
              .then((r) => r.json())
              .then((data: any) => {
                const items: any[] = data?.data || [];
                if (items.length === 0) return null;
                return {
                  source: 'jina-search' as const,
                  results: items.slice(0, 5).map((item: any) => ({
                    title: item.title || '',
                    url: item.url || '',
                    snippet: extractSnippet(item),
                  })),
                };
              })
              .catch(() => null);

            const ddgSearch = searchDDG()
              .then((results) =>
                results.length > 0
                  ? { source: 'ddg-fallback' as const, results }
                  : null,
              )
              .catch(() => null);

            // Take first non-null result; prefer jina if both arrive quickly
            const winner = await Promise.any([
              jinaSearch.then((r) => r ?? Promise.reject()),
              ddgSearch.then((r) => r ?? Promise.reject()),
            ]).catch(() => null);

            if (winner) {
              logger.info(
                { query, count: winner.results.length, source: winner.source },
                'webSearch results',
              );
              return { results: winner.results };
            }
            return {
              results: [],
              note: 'No results — try webFetch with a specific URL',
            };
          }

          // No API key: DDG only
          const results = await searchDDG();
          if (results.length === 0) {
            logger.warn({ query }, 'webSearch: no results from DDG');
            return {
              results: [],
              note: 'No results — try webFetch with a specific URL',
            };
          }
          logger.info(
            { query, count: results.length, source: 'ddg-fallback' },
            'webSearch results',
          );
          return { results };
        } catch (err: any) {
          logger.warn({ query, err: (err as any).message }, 'webSearch failed');
          return { error: (err as any).message };
        }
      },
    }),

    ...(input.isMain
      ? {
          gmailSearch: tool({
            description:
              'Search or list Gmail emails. Returns sender, subject, date and body preview for each result. ' +
              'Use simple keywords to search broadly (e.g. "digitalocean" searches from, subject AND body). ' +
              'Never invent a full email address — use just the domain or keyword. ' +
              'Examples: "digitalocean", "is:unread", "from:amazon", "subject:facture". ' +
              'The body preview (up to 800 chars) is included — you do NOT need to call gmailRead for basic summaries.',
            inputSchema: z.object({
              query: z
                .string()
                .default('in:inbox')
                .describe('Gmail search query. Default: "in:inbox"'),
              maxResults: z
                .number()
                .default(5)
                .describe('Max number of results'),
            }),
            execute: async ({ query: rawQuery, maxResults }) => {
              const query = preprocessGmailQuery(rawQuery);
              logger.info(
                { rawQuery, query, maxResults },
                'gmailSearch called',
              );
              const gmail = createGmailClient();
              if (!gmail) return { error: 'Gmail not configured' };
              try {
                const list = await gmail.users.messages.list({
                  userId: 'me',
                  q: query,
                  maxResults,
                });
                const messages = list.data.messages || [];
                if (messages.length === 0) return { results: [] };

                const extractBody = (part: any): string => {
                  if (part.mimeType === 'text/plain' && part.body?.data) {
                    return Buffer.from(part.body.data, 'base64').toString(
                      'utf-8',
                    );
                  }
                  if (part.parts) {
                    for (const p of part.parts) {
                      const text = extractBody(p);
                      if (text) return text;
                    }
                  }
                  return '';
                };

                const results = await Promise.all(
                  messages.map(async (m) => {
                    const msg = await gmail.users.messages.get({
                      userId: 'me',
                      id: m.id!,
                      format: 'full',
                    });
                    const h = (name: string) =>
                      msg.data.payload?.headers?.find(
                        (hdr) => hdr.name?.toLowerCase() === name.toLowerCase(),
                      )?.value || '';
                    const body = extractBody(msg.data.payload || {}).slice(
                      0,
                      800,
                    );
                    return {
                      id: m.id,
                      from: h('From'),
                      subject: h('Subject'),
                      date: h('Date'),
                      body: body || msg.data.snippet || '',
                    };
                  }),
                );
                logger.info({ count: results.length }, 'gmailSearch results');
                return { results };
              } catch (err: any) {
                logger.error({ err: err.message }, 'gmailSearch error');
                return { error: err.message };
              }
            },
          }),

          gmailSend: tool({
            description: 'Send a new email via Gmail.',
            inputSchema: z.object({
              to: z.string().describe('Recipient email address'),
              subject: z.string().describe('Email subject'),
              body: z.string().describe('Email body (plain text)'),
            }),
            execute: async ({ to, subject, body }) => {
              const gmail = createGmailClient();
              if (!gmail) return { error: 'Gmail not configured' };
              logger.info({ to, subject }, 'gmailSend called');
              try {
                const raw = Buffer.from(
                  [
                    `To: ${to}`,
                    `Subject: ${subject}`,
                    'Content-Type: text/plain; charset=utf-8',
                    '',
                    body,
                  ].join('\r\n'),
                )
                  .toString('base64')
                  .replace(/\+/g, '-')
                  .replace(/\//g, '_')
                  .replace(/=+$/, '');

                const res = await gmail.users.messages.send({
                  userId: 'me',
                  requestBody: { raw },
                });
                logger.info(
                  { to, subject, id: res.data.id },
                  'gmailSend success',
                );
                return { success: true, to, subject };
              } catch (err: any) {
                logger.error(
                  { to, subject, error: err.message },
                  'gmailSend failed',
                );
                return { error: err.message };
              }
            },
          }),

          gmailDelete: tool({
            description:
              'Search for a Gmail email and move it to trash (recoverable for 30 days). ' +
              'Be SPECIFIC with your keywords to match exactly one email (e.g. "edf tempo 16 mars", not just "edf"). ' +
              'If exactly one email matches, it is trashed. ' +
              'If multiple match, nothing is deleted — the tool returns the list so you can refine your query with more specific keywords or a date.',
            inputSchema: z.object({
              query: z
                .string()
                .describe(
                  'Specific Gmail search keywords to identify the email to delete',
                ),
            }),
            execute: async ({ query: rawQuery }) => {
              const query = preprocessGmailQuery(rawQuery);
              const gmail = createGmailClient();
              if (!gmail) return { error: 'Gmail not configured' };
              logger.info({ rawQuery, query }, 'gmailDelete called');
              try {
                const list = await gmail.users.messages.list({
                  userId: 'me',
                  q: query,
                  maxResults: 5,
                });
                const messages = list.data.messages || [];
                if (messages.length === 0)
                  return { error: 'No email found matching this query.' };
                if (messages.length > 1) {
                  const summaries = await Promise.all(
                    messages.map(async (m) => {
                      const msg = await gmail.users.messages.get({
                        userId: 'me',
                        id: m.id!,
                        format: 'metadata',
                        metadataHeaders: ['From', 'Subject', 'Date'],
                      });
                      const h = (name: string) =>
                        msg.data.payload?.headers?.find(
                          (hdr) =>
                            hdr.name?.toLowerCase() === name.toLowerCase(),
                        )?.value || '';
                      return {
                        from: h('From'),
                        subject: h('Subject'),
                        date: h('Date'),
                      };
                    }),
                  );
                  return {
                    tooMany: true,
                    count: summaries.length,
                    matches: summaries,
                    hint: 'Add a date or more keywords to match exactly one email.',
                  };
                }
                const id = messages[0].id!;
                const msg = await gmail.users.messages.get({
                  userId: 'me',
                  id,
                  format: 'metadata',
                  metadataHeaders: ['From', 'Subject', 'Date'],
                });
                const h = (name: string) =>
                  msg.data.payload?.headers?.find(
                    (hdr) => hdr.name?.toLowerCase() === name.toLowerCase(),
                  )?.value || '';
                await gmail.users.messages.trash({ userId: 'me', id });
                logger.info(
                  { id, subject: h('Subject') },
                  'gmailDelete success',
                );
                return {
                  success: true,
                  trashed: {
                    from: h('From'),
                    subject: h('Subject'),
                    date: h('Date'),
                  },
                };
              } catch (err: any) {
                logger.error({ query, err: err.message }, 'gmailDelete error');
                return { error: err.message };
              }
            },
          }),

          setModel: tool({
            description:
              'Change the Ollama model used by this agent. Omit the model parameter to list available models. The change takes effect immediately and persists across restarts.',
            inputSchema: z.object({
              model: z
                .string()
                .optional()
                .describe(
                  'Model name to switch to (e.g. "llama3.2:3b"). Omit to list available models.',
                ),
            }),
            execute: async ({ model }) => {
              let availableModels: string[] = [];
              try {
                const res = await fetch(`${OLLAMA_BASE_URL}/api/tags`, {
                  signal: AbortSignal.timeout(5_000),
                });
                const data = (await res.json()) as {
                  models?: Array<{ name: string }>;
                };
                availableModels = (data.models || []).map((m) => m.name);
              } catch (err: any) {
                return { error: `Cannot reach Ollama: ${err.message}` };
              }

              if (!model) {
                return { currentModel, availableModels };
              }

              if (!availableModels.includes(model)) {
                return {
                  error: `Model "${model}" not found in Ollama`,
                  availableModels,
                };
              }

              currentModel = model;
              writeEnvFile('OLLAMA_MODEL', model);

              return { success: true, currentModel, availableModels };
            },
          }),
        }
      : {}),
  };
}

// ─── Main entry point ─────────────────────────────────────────────────────────

export async function runAgentEngine(
  group: RegisteredGroup,
  input: ContainerInput,
  mounts: VolumeMount[],
  callbacks: AgentCallbacks,
  onOutput?: (output: ContainerOutput) => Promise<void>,
): Promise<ContainerOutput> {
  // 1. Check Ollama availability
  try {
    await fetch(`${OLLAMA_BASE_URL}/api/tags`, {
      signal: AbortSignal.timeout(3_000),
    });
  } catch {
    const error = `Ollama indisponible à ${OLLAMA_BASE_URL}`;
    logger.error({ group: group.name, url: OLLAMA_BASE_URL }, error);
    const out: ContainerOutput = { status: 'error', result: null, error };
    if (onOutput) await onOutput(out);
    return out;
  }

  // 2. Start / retrieve sandbox container
  let containerName: string;
  try {
    containerName = await ensureSandbox(group, mounts);
  } catch (err: any) {
    const error = `Sandbox failed to start: ${err.message}`;
    logger.error({ group: group.name }, error);
    const out: ContainerOutput = { status: 'error', result: null, error };
    if (onOutput) await onOutput(out);
    return out;
  }

  // 3. Build prompt components
  const ollama = createOllama({ baseURL: `${OLLAMA_BASE_URL}/api` });
  const systemPrompt = buildSystemPrompt(
    group,
    input.isMain,
    callbacks,
    input.isScheduledTask,
  );
  const tools = buildTools(containerName, group, input, callbacks);

  logger.info(
    { group: group.name, model: currentModel, containerName },
    'Starting Ollama agent',
  );

  // 4. Run the agentic loop
  const history = input.isScheduledTask
    ? []
    : getConversationHistory(group.folder, CONVERSATION_HISTORY_TURNS);

  const totalChars =
    systemPrompt.length +
    history.reduce((acc, m) => acc + m.content.length, 0) +
    input.prompt.length;
  if (totalChars > 80_000) {
    logger.warn(
      {
        group: group.name,
        totalChars,
        historyTurns: history.length / 2,
      },
      'Context window approaching limit — consider reducing CONVERSATION_HISTORY_TURNS',
    );
  }

  try {
    // Pass image as raw base64 string (no "data:" prefix).
    // - Uint8Array/Buffer → SDK stores as Uint8Array → Ollama provider serializes as JSON object ✗
    // - "data:..." URL string → SDK sees a valid URL → tries to download it → DownloadError ✗
    // - Raw base64 string → new URL() throws → SDK stores as plain string → provider sends
    //   images:["base64string"] to Ollama ✓
    const userMessage = input.imageBase64
      ? {
          role: 'user' as const,
          content: [
            { type: 'text' as const, text: input.prompt },
            {
              type: 'image' as const,
              image: input.imageBase64,
              mimeType: (input.imageMimeType ?? 'image/jpeg') as any,
            },
          ],
        }
      : { role: 'user' as const, content: input.prompt };

    const result = await generateText({
      model: ollama(currentModel),
      system: systemPrompt,
      messages: [...history, userMessage],
      tools,
      stopWhen: stepCountIs(MAX_STEPS),
      onStepFinish: async (step) => {
        if (step.text?.trim()) {
          logger.debug(
            { group: group.name, stepText: step.text.slice(0, 100) },
            'Agent step text',
          );
        }
        if (step.toolCalls?.length) {
          for (const tc of step.toolCalls) {
            logger.info(
              { group: group.name, tool: tc!.toolName, args: (tc as any).args },
              'Agent tool call',
            );
          }
        }
        if (step.toolResults?.length) {
          for (const tr of step.toolResults) {
            logger.info(
              {
                group: group.name,
                tool: tr!.toolName,
                result: JSON.stringify((tr as any).result).slice(0, 200),
              },
              'Agent tool result',
            );
          }
        }
      },
    });

    // 5. Persist conversation turn (skip scheduled tasks — isolated context)
    if (result.text && !input.isScheduledTask) {
      const toolSummary = result.steps
        .flatMap((s) => s.toolCalls ?? [])
        .map((tc) => tc!.toolName)
        .join(', ');
      storeConversationTurn(
        group.folder,
        input.prompt,
        result.text,
        toolSummary || undefined,
      );
    }

    // 6. Archive conversation
    const convDir = path.join(
      resolveGroupFolderPath(group.folder),
      'conversations',
    );
    fs.mkdirSync(convDir, { recursive: true });
    const ts = new Date().toISOString().replace(/[:.]/g, '-');
    fs.writeFileSync(
      path.join(convDir, `${ts}.md`),
      `# ${new Date().toISOString()}\n\n**Prompt:**\n${input.prompt}\n\n**Response:**\n${result.text}\n`,
    );

    logger.info(
      { group: group.name, steps: result.steps.length, model: currentModel },
      'Agent completed',
    );

    const output: ContainerOutput = {
      status: 'success',
      result: result.text || null,
    };
    if (onOutput) await onOutput(output);
    return output;
  } catch (err: any) {
    logger.error({ group: group.name, err }, 'Agent engine error');
    const error = `Agent error: ${err.message}`;
    const out: ContainerOutput = { status: 'error', result: null, error };
    if (onOutput) await onOutput(out);
    return out;
  }
}
