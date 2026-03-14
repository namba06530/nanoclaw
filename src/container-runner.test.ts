import { describe, it, expect, vi } from 'vitest';

// Mock config
vi.mock('./config.js', () => ({
  DATA_DIR: '/tmp/nanoclaw-test-data',
  GROUPS_DIR: '/tmp/nanoclaw-test-groups',
  TIMEZONE: 'America/Los_Angeles',
}));

// Mock logger
vi.mock('./logger.js', () => ({
  logger: {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

// Mock fs
vi.mock('fs', async () => {
  const actual = await vi.importActual<typeof import('fs')>('fs');
  return {
    ...actual,
    default: {
      ...actual,
      existsSync: vi.fn(() => false),
      mkdirSync: vi.fn(),
      writeFileSync: vi.fn(),
      readFileSync: vi.fn(() => ''),
      readdirSync: vi.fn(() => []),
      statSync: vi.fn(() => ({ isDirectory: () => false })),
    },
  };
});

// Mock mount-security
vi.mock('./mount-security.js', () => ({
  validateAdditionalMounts: vi.fn(() => []),
}));

// Mock container-runtime
vi.mock('./container-runtime.js', () => ({
  readonlyMountArgs: vi.fn((h: string, c: string) => [
    `--mount`,
    `type=bind,src=${h},dst=${c},readonly`,
  ]),
}));

// Mock agent-engine — the core of the new architecture
vi.mock('./agent-engine.js', () => ({
  runAgentEngine: vi.fn(),
  stopAllSandboxes: vi.fn(),
  ensureSandbox: vi.fn(),
  stopSandbox: vi.fn(),
}));

import { runContainerAgent } from './container-runner.js';
import { runAgentEngine } from './agent-engine.js';
import type { RegisteredGroup } from './types.js';

const testGroup: RegisteredGroup = {
  name: 'Test Group',
  folder: 'test-group',
  trigger: '@Andy',
  added_at: new Date().toISOString(),
};

const testInput = {
  prompt: 'Hello',
  groupFolder: 'test-group',
  chatJid: 'test@g.us',
  isMain: false,
};

describe('container-runner (Ollama engine delegation)', () => {
  it('delegates to runAgentEngine and returns its result on success', async () => {
    const onOutput = vi.fn(async () => {});
    const expectedOutput = {
      status: 'success' as const,
      result: 'Here is my response',
    };
    vi.mocked(runAgentEngine).mockResolvedValueOnce(expectedOutput);

    const result = await runContainerAgent(
      testGroup,
      testInput,
      () => {},
      onOutput,
    );

    expect(vi.mocked(runAgentEngine)).toHaveBeenCalledOnce();
    expect(result.status).toBe('success');
    expect(result.result).toBe('Here is my response');
  });

  it('returns error when runAgentEngine fails', async () => {
    const onOutput = vi.fn(async () => {});
    vi.mocked(runAgentEngine).mockResolvedValueOnce({
      status: 'error',
      result: null,
      error: 'Ollama indisponible',
    });

    const result = await runContainerAgent(
      testGroup,
      testInput,
      () => {},
      onOutput,
    );

    expect(result.status).toBe('error');
    expect(result.error).toContain('Ollama');
  });

  it('calls onProcess callback with null proc and container name', async () => {
    vi.mocked(runAgentEngine).mockResolvedValueOnce({
      status: 'success',
      result: null,
    });
    const onProcess = vi.fn();

    await runContainerAgent(testGroup, testInput, onProcess, undefined);

    expect(onProcess).toHaveBeenCalledOnce();
    const [proc, name] = onProcess.mock.calls[0];
    expect(proc).toBeNull();
    expect(name).toContain('nanoclaw-sandbox');
  });
});
