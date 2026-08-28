export type JobStatus = 'QUEUED' | 'RUNNING' | 'SUCCESS' | 'FAILED';

export interface Job { job_id: string; job_type: string; status: JobStatus; progress: number; input_files: string[]; output_files: string[]; started_at: string | null; finished_at: string | null; error_message: string | null; log_path: string; }
export interface UploadResult { upload_id: string; job_id: string; filename: string; stored_path: string; csv_files: string[]; headers: string[]; field_mapping: Record<string, string | null>; mapping_status: 'READY' | 'NEEDS_CONFIRMATION'; created_at: string; }
export interface Dataset { dataset_id: string; dataset_name: string; source_type: string; source_files: string[]; date_range: { start: string; end: string }; store_count: number; item_count: number; row_count: number; monthly_parquet_path: string; active_store_parquet_path: string; feature_parquet_path: string; has_active_store: boolean; has_diff_features: boolean; has_cross_store_features: boolean; created_at: string; status: string; }
export interface PredictionRun { prediction_run_id: string; dataset_id: string; model_ids: string[]; observation_month: string | null; store_ids: string[] | null; job_id: string; status: JobStatus; prediction_dir: string | null; created_at: string; error_message: string | null; }
export interface FilteredPredictionSummary { filters: { model_id: string | null; site_no: string | null; mc: string | null }; prediction_total: number; predicted_nonzero_book_count: number; mc_counts: Record<string, number>; predicted_20_plus_book_count: number; store_count: number; item_count: number; row_count: number; }
export interface PredictionSummary { prediction_run_id: string; dataset_id: string; observation_month: string; model_ids: string[]; model_summaries: Record<string, { prediction_total: number; predicted_nonzero_book_count: number; mc_counts: Record<string, number>; predicted_20_plus_book_count: number; store_count: number; item_count: number; row_count: number }>; filtered_summary: FilteredPredictionSummary; }
export interface NotificationRecord { notification_id: string; type: 'TEST' | 'PREDICTION_SUCCESS' | 'PREDICTION_FAILED' | 'EXPORT_SUCCESS'; target_email: string; status: 'SUCCESS' | 'FAILED'; related_run_id: string | null; created_at: string; error_message: string | null; }

export const apiBaseUrl = () => (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  if (!apiBaseUrl()) throw new Error('VITE_API_BASE_URL is required');
  const response = await fetch(`${apiBaseUrl()}${path}`, init ?? {});
  if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.detail || `请求失败（${response.status}）`); }
  return response.json() as Promise<T>;
}

export const getHealth = () => request<{ status: string }>('/api/health');
export const getJobs = () => request<{ items: Job[] }>('/api/jobs');
export const getJob = (id: string) => request<Job>(`/api/jobs/${id}`);
export const getDatasets = () => request<{ items: Dataset[] }>('/api/datasets');
export const getDataset = (id: string) => request<Dataset>(`/api/datasets/${id}`);
export async function uploadFile(file: File) { const form = new FormData(); form.append('file', file); return request<UploadResult>('/api/uploads', { method: 'POST', body: form }); }
export const processDataset = (upload_id: string, dataset_name: string) => request<{ job_id: string; dataset_id: string }>('/api/datasets/process', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ upload_id, dataset_name }) });
export const createPrediction = (payload: { dataset_id: string; model_ids: string[]; observation_month?: string; store_ids?: string[] }) => request<{ prediction_run_id: string; job_id: string; status: JobStatus }>('/api/predictions', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
export const getPredictions = () => request<{ items: PredictionRun[] }>('/api/predictions');
export const getPrediction = (runId: string) => request<PredictionRun>(`/api/predictions/${runId}`);
export const getPredictionSummary = (runId: string, filters: Pick<ResultFilters, 'modelId' | 'siteNo' | 'mc'> = {}) => {
  const params = new URLSearchParams();
  if (filters.modelId) params.set('model_id', filters.modelId);
  if (filters.siteNo) params.set('site_no', filters.siteNo);
  if (filters.mc) params.set('mc', filters.mc);
  const query = params.toString();
  return request<PredictionSummary>(`/api/predictions/${runId}/summary${query ? `?${query}` : ''}`);
};
export interface ResultFilters { modelId?: string; siteNo?: string; mc?: string; page?: number; pageSize?: number; }
export interface PagedResults { items: Record<string, unknown>[]; total: number; page: number; page_size: number; kind: string; }
export const getPredictionResults = (runId: string, kind: 'predictions' | 'top_books' | 'store_summary', filters: ResultFilters = {}) => {
  const params = new URLSearchParams({ kind });
  if (filters.modelId) params.set('model_id', filters.modelId);
  if (filters.siteNo) params.set('site_no', filters.siteNo);
  if (filters.mc) params.set('mc', filters.mc);
  if (filters.page) params.set('page', String(filters.page));
  if (filters.pageSize) params.set('page_size', String(filters.pageSize));
  return request<PagedResults>(`/api/predictions/${runId}/results?${params.toString()}`);
};
export const getStorePredictionSummary = (runId: string, siteNo: string, modelId: string) => request<{ prediction_total: number; pred_nonzero_count: number; pred_mc3_count: number; pred_mc4_count: number; item_count: number }>(`/api/predictions/${runId}/store/${encodeURIComponent(siteNo)}/summary?model_id=${encodeURIComponent(modelId)}`);
export const predictionDownloadUrl = (runId: string) => `${apiBaseUrl()}/api/predictions/${runId}/download-parquet`;
export const predictionExcelUrl = (runId: string, options: { modelId?: string; siteNo?: string; mc?: string; includePredictions?: boolean; topN?: number } = {}) => {
  const params = new URLSearchParams();
  if (options.modelId) params.set('model_id', options.modelId);
  if (options.siteNo) params.set('site_no', options.siteNo);
  if (options.mc) params.set('mc', options.mc);
  if (options.includePredictions) params.set('include_predictions', 'true');
  if (options.topN) params.set('top_n', String(options.topN));
  return `${apiBaseUrl()}/api/predictions/${runId}/export-excel?${params.toString()}`;
};
export const sendTestEmail = (target_email: string) => request<NotificationRecord>('/api/notifications/test-email', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ target_email }) });
export const getNotifications = () => request<{ items: NotificationRecord[] }>('/api/notifications');
export const sendPredictionNotification = (runId: string, payload: { target_email: string; model_id?: string; include_excel_link?: boolean; notification_type?: 'PREDICTION_SUCCESS' | 'EXPORT_SUCCESS'; site_no?: string; mc?: string }) => request<NotificationRecord>(`/api/predictions/${runId}/notify`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
