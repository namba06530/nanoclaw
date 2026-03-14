/**
 * Agent Engine for NanoClaw — Ollama-powered (Vercel AI SDK v6)
 *
 * Replaces the Anthropic Claude Code SDK. The LLM (Ollama) runs on the host
 * with GPU access. Bash/file tools execute via `docker exec` into a minimal
 * per-group sandbox container for isolation.
 */
import { exec as execCallback, spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import { promisify } from 'util';

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
  getConversationHistory,
  storeConversationTurn,
} from './db.js';
import { resolveGroupFolderPath } from './group-folder.js';
import { logger } from './logger.js';
import { RegisteredGroup, ScheduledTask } from './types.js';

const execAsync = promisify(execCallback);

const MAX_STEPS = AGENT_MAX_STEPS;
const BASH_TIMEOUT_MS = 30_000;

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

  // Instructions for brevity (local models tend to be verbose)
  parts.push(
    '\n\n## Response Instructions\n' +
      'Be concise. Use the language of the user. ' +
      'Do not add unnecessary disclaimers or repetition. ' +
      'For messaging apps: keep responses short unless detail is explicitly requested.',
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
      description: 'Fetch content from a URL',
      inputSchema: z.object({
        url: z.string().describe('URL to fetch'),
        format: z
          .enum(['text', 'json'])
          .default('text')
          .describe('Response format'),
      }),
      execute: async ({ url, format }) => {
        try {
          const res = await fetch(url, {
            headers: { 'User-Agent': 'NanoClaw/1.0' },
            signal: AbortSignal.timeout(15_000),
          });
          const text = await res.text();
          if (format === 'json') {
            try {
              return { content: JSON.parse(text), status: res.status };
            } catch {
              return { content: text.slice(0, 50_000), status: res.status };
            }
          }
          return { content: text.slice(0, 50_000), status: res.status };
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
    { group: group.name, model: OLLAMA_MODEL, containerName },
    'Starting Ollama agent',
  );

  // 4. Run the agentic loop
  const history = input.isScheduledTask
    ? []
    : getConversationHistory(group.folder, CONVERSATION_HISTORY_TURNS);

  try {
    const result = await generateText({
      model: ollama(OLLAMA_MODEL),
      system: systemPrompt,
      messages: [...history, { role: 'user', content: input.prompt }],
      tools,
      stopWhen: stepCountIs(MAX_STEPS),
      onStepFinish: async (step) => {
        // Log intermediate steps for debugging (don't send to user — final onOutput handles that)
        if (step.text?.trim()) {
          logger.debug(
            { group: group.name, stepText: step.text.slice(0, 100) },
            'Agent step text',
          );
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
      { group: group.name, steps: result.steps.length, model: OLLAMA_MODEL },
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
