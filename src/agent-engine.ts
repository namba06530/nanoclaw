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
  context_mode?: 'group' | 'isolated';
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

  // Scheduled tasks context
  const allTasks = callbacks.getAllTasks();
  const tasks = isMain
    ? allTasks
    : allTasks.filter((t) => t.group_folder === group.folder);
  if (tasks.length > 0) {
    parts.push('\n\n## Scheduled Tasks\n' + JSON.stringify(tasks, null, 2));
  }

  // Available groups (main only)
  if (isMain) {
    const groups = callbacks.getAvailableGroups();
    parts.push('\n\n## Available Groups\n' + JSON.stringify(groups, null, 2));
  }

  // Instructions for brevity and tool usage
  parts.push(
    '\n\n## Response Instructions\n' +
      'Be concise. Use the language of the user. ' +
      'Do not add unnecessary disclaimers or repetition. ' +
      'For messaging apps: keep responses short unless detail is explicitly requested.\n\n' +
      '## CRITICAL: Tool Usage\n' +
      'You MUST use your tools to perform actions. NEVER pretend to have performed an action without calling the appropriate tool. ' +
      'Every email send requires a gmailSend tool call. Every file read requires a readFile tool call. ' +
      'If a user asks you to do something and you have a tool for it, you MUST call the tool — do NOT simulate or hallucinate the result.\n\n' +
      '## Fichiers joints\n' +
      'Quand un message commence par `Fichier joint "nom" :`, le contenu du fichier a déjà été extrait et t\'est fourni directement dans le message. ' +
      "Lis et utilise ce contenu directement — pas besoin d'outil. " +
      "Si le contenu dit \"[PDF scanné ...]\", informe l'utilisateur que l'OCR n'est pas disponible pour ce document scanné.",
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
          .enum(['group', 'isolated'])
          .default('group')
          .describe(
            'group: uses conversation context, isolated: fresh session',
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

    deleteTask: tool({
      description:
        'Permanently delete a scheduled task by its ID. Use when the user wants to cancel or remove a task.',
      inputSchema: z.object({
        task_id: z.string().describe('The task ID to delete'),
      }),
      execute: async ({ task_id }) => {
        try {
          const all = callbacks.getAllTasks();
          const task = all.find((t) => t.id === task_id);
          if (!task) return { error: `Task "${task_id}" not found` };
          callbacks.deleteTask(task_id);
          return { success: true, deleted: task_id };
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
        const searchDDG = async () => {
          const jinaUrl = `https://r.jina.ai/https://html.duckduckgo.com/html/?q=${encoded}`;
          const res = await fetch(jinaUrl, {
            headers: { 'User-Agent': 'NanoClaw/1.0', 'X-Retain-Images': 'none' },
            signal: AbortSignal.timeout(15_000),
          });
          const markdown = await res.text();
          const results: Array<{ title: string; url: string }> = [];
          const linkPattern =
            /\[([^\]]{5,})\]\(https:\/\/duckduckgo\.com\/l\/\?uddg=([^&)]+)[^)]*\)/g;
          let match;
          while ((match = linkPattern.exec(markdown)) !== null && results.length < 5) {
            const title = match[1].trim();
            const realUrl = decodeURIComponent(match[2]);
            if (realUrl.startsWith('http') && !realUrl.includes('duckduckgo.com')) {
              results.push({ title, url: realUrl });
            }
          }
          return results;
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
              logger.info({ query, count: winner.results.length, source: winner.source }, 'webSearch results');
              return { results: winner.results };
            }
            return { results: [], note: 'No results — try webFetch with a specific URL' };
          }

          // No API key: DDG only
          const results = await searchDDG();
          if (results.length === 0) {
            logger.warn({ query }, 'webSearch: no results from DDG');
            return { results: [], note: 'No results — try webFetch with a specific URL' };
          }
          logger.info({ query, count: results.length, source: 'ddg-fallback' }, 'webSearch results');
          return { results };
        } catch (err: any) {
          logger.warn({ query, err: (err as any).message }, 'webSearch failed');
          return { error: (err as any).message };
        }
      },
    }),

    gmailSearch: tool({
      description:
        'Search or list Gmail emails. Returns a list with sender, subject, date and snippet. Use Gmail search syntax for the query (e.g. "is:unread", "from:boss@company.com", "subject:invoice").',
      inputSchema: z.object({
        query: z
          .string()
          .default('in:inbox')
          .describe('Gmail search query. Default: "in:inbox"'),
        maxResults: z.number().default(10).describe('Max number of results'),
      }),
      execute: async ({ query, maxResults }) => {
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

          const results = await Promise.all(
            messages.map(async (m) => {
              const msg = await gmail.users.messages.get({
                userId: 'me',
                id: m.id!,
                format: 'metadata',
                metadataHeaders: ['From', 'Subject', 'Date'],
              });
              const h = (name: string) =>
                msg.data.payload?.headers?.find(
                  (hdr) => hdr.name?.toLowerCase() === name.toLowerCase(),
                )?.value || '';
              return {
                id: m.id,
                from: h('From'),
                subject: h('Subject'),
                date: h('Date'),
                snippet: msg.data.snippet || '',
              };
            }),
          );
          return { results };
        } catch (err: any) {
          return { error: err.message };
        }
      },
    }),

    gmailRead: tool({
      description: 'Read the full content of a Gmail email by its message ID.',
      inputSchema: z.object({
        messageId: z.string().describe('Gmail message ID'),
      }),
      execute: async ({ messageId }) => {
        const gmail = createGmailClient();
        if (!gmail) return { error: 'Gmail not configured' };
        try {
          const msg = await gmail.users.messages.get({
            userId: 'me',
            id: messageId,
            format: 'full',
          });
          const h = (name: string) =>
            msg.data.payload?.headers?.find(
              (hdr) => hdr.name?.toLowerCase() === name.toLowerCase(),
            )?.value || '';

          // Extract text body
          let body = '';
          const extractBody = (part: any): string => {
            if (part.mimeType === 'text/plain' && part.body?.data) {
              return Buffer.from(part.body.data, 'base64').toString('utf-8');
            }
            if (part.parts) {
              for (const p of part.parts) {
                const text = extractBody(p);
                if (text) return text;
              }
            }
            return '';
          };
          body = extractBody(msg.data.payload || {});

          return {
            id: messageId,
            from: h('From'),
            to: h('To'),
            subject: h('Subject'),
            date: h('Date'),
            body: body.slice(0, 4000),
          };
        } catch (err: any) {
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
          logger.info({ to, subject, id: res.data.id }, 'gmailSend success');
          return { success: true, to, subject };
        } catch (err: any) {
          logger.error({ to, subject, error: err.message }, 'gmailSend failed');
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
  const systemPrompt = buildSystemPrompt(group, input.isMain, callbacks);
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
      storeConversationTurn(group.folder, input.prompt, result.text);
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
