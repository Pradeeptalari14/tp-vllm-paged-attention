/**
 * vLLM Continuous Batcher & Chunked Prefill Co-Scheduler
 * Iteration-level scheduling interleaving prefill chunks with decode steps.
 */

export interface InferenceRequest {
  id: string;
  promptTokens: number[];
  generatedTokens: number[];
  maxTokens: number;
  phase: 'PREFILL' | 'DECODE' | 'COMPLETED';
}

export class ContinuousBatcher {
  private activeQueue: InferenceRequest[] = [];
  private chunkedPrefillLimit = 512;

  public submit(request: InferenceRequest) {
    this.activeQueue.push(request);
  }

  public stepIteration(): { prefillChunks: number; decodes: number; completed: number } {
    let prefillChunks = 0;
    let decodes = 0;
    let completed = 0;

    for (const req of this.activeQueue) {
      if (req.phase === 'PREFILL') {
        prefillChunks++;
        req.phase = 'DECODE';
      } else if (req.phase === 'DECODE') {
        decodes++;
        req.generatedTokens.push(Math.floor(Math.random() * 32000));
        if (req.generatedTokens.length >= req.maxTokens) {
          req.phase = 'COMPLETED';
          completed++;
        }
      }
    }

    this.activeQueue = this.activeQueue.filter(r => r.phase !== 'COMPLETED');
    return { prefillChunks, decodes, completed };
  }
}
