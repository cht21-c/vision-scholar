export interface Paper {
  id: string; title: string; authors: string; year: number | null
  source_url: string; sha256: string; page_count: number; chunk_count: number
}
export interface Evidence {
  id: string; text: string; title: string; paper_id?: string; page?: number
  section?: string; kind?: string; path?: string; start?: number; end?: number
}
export interface ResearchRun {
  id: string; session_id: string; prompt: string; paper_ids: string[]
  harness: string; retriever: string
  status: string; mode: string; role: string; answer: string; citations: Evidence[]
  error?: string; created_at: string; usage: { provider?: string; model?: string }
  verification: { status?: string; scope?: string }
}
export interface Session {
  id: string; title: string; updated_at: string; harness?: string; retriever?: string
}
export interface Note { id: string; session_id: string; title: string; content: string }
export interface Epoch {
  epoch: number; train_loss: number; val_loss: number
  train_accuracy: number; val_accuracy: number
}
export interface Variant {
  variant: string; test_accuracy: number; validation_accuracy: number
  test_macro_f1: number; selected_C: number; confusion_matrix: number[][]
}
export interface Experiment {
  id: string; kind: string; created_at: string; config: { seed: number; pca_components: number }
  result: {
    dataset: string; seed: number; elapsed_seconds: number; code_sha256: string
    split?: { train: number; validation: number; test: number }; results?: Variant[]
    training_log?: Epoch[]; training_log_source?: string; limitation: string
    patch_shape?: number[]; output_shape?: number[]; attention_preview?: number[][]
    row_sums_min?: number; row_sums_max?: number; token_count?: number
  }
}
export interface LogResult {
  best_accuracy_epoch: number; best_val_accuracy: number; best_loss_epoch: number
  best_val_loss: number; last_generalization_gap: number; signals: string[]; records: Epoch[]
}
export interface Study {
  id: string; status: string; created_at: string; error?: string
  plan: { config: { seeds: number[]; pca_components: number; c_candidates: number[]
    noise_sigma: number; occlusion_size: number }; selection: string; limitations: string }
  result?: {
    summary: { condition: string; method: string; mean_accuracy: number; std_accuracy: number }[]
    paired_comparisons: { seed: number; condition: string; pca_minus_pixels: number
      paired_bootstrap_95_low: number; paired_bootstrap_95_high: number }[]
    elapsed_seconds: number; evidence_files: Record<string, string>; limitations: string
  }
}
export interface ReportCase {
  id: string; question?: string; name?: string; passed?: boolean
  status?: string; answer?: string; expected?: string; reason?: string
  citations?: string[]; elapsed_seconds?: number
}
export interface Report {
  file: string; name?: string; kind: string; created_at: string; model?: string
  summary?: { passed: number; total: number; description?: string }
  cases?: ReportCase[]; run?: ResearchRun; elapsed_seconds?: number
}
export interface Bootstrap {
  model: { provider: string; model: string; available: boolean; mock: boolean; runtime: string
    harnesses: string[]; retrievers: string[] }
  papers: Paper[]; sessions: Session[]; notes: Note[]; experiments: Experiment[]; reports: Report[]
  studies: Study[]
  stats: { papers: number; pages: number; chunks: number }
}
export interface RunEvent {
  seq: number; run_id: string; type: string; created_at: string
  data: {
    text?: string; message?: string; label?: string; role?: string; name?: string
    input?: Record<string, unknown>; output?: string; is_error?: boolean
    items?: Evidence[]; has_tools?: boolean; status?: string; error?: string
  }
}
export async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, options)
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(typeof error.detail === 'string' ? error.detail : '输入格式不正确，请检查后重试')
  }
  return response.json()
}
export function post<T>(path: string, body?: unknown) {
  return api<T>(path, { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body) })
}
export function upload<T>(path: string, file: File) {
  const form = new FormData()
  form.append('file', file)
  return api<T>(path, { method: 'POST', body: form })
}
export const percent = (value: number) => `${(value * 100).toFixed(2)}%`
export const activeStatus = (status: string) => ['queued', 'running'].includes(status)
