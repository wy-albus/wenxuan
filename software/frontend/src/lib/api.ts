export type JobStatus = 'QUEUED' | 'RUNNING' | 'SUCCESS' | 'FAILED';

export interface Job { job_id: string; job_type: string; status: JobStatus; progress: number; input_files: string[]; output_files: string[]; started_at: string | null; finished_at: string | null; error_message: string | null; log_path: string; }
export interface UploadResult { upload_id: string; job_id: string; filename: string; stored_path: string; csv_files: string[]; headers: string[]; field_mapping: Record<string, string | null>; mapping_status: 'READY' | 'NEEDS_CONFIRMATION'; created_at: string; }
export interface UploadRecord extends UploadResult { file_size_bytes: number | null; processing_status: '已处理' | '未处理'; related_dataset: { dataset_id: string; dataset_name: string } | null; }
export interface Dataset { dataset_id: string; dataset_name: string; source_type: string; source_files: string[]; date_range: { start: string; end: string }; store_count: number; item_count: number; row_count: number; monthly_parquet_path: string; active_store_parquet_path: string; feature_parquet_path: string; has_active_store: boolean; has_diff_features: boolean; has_cross_store_features: boolean; created_at: string; status: string; }
export interface PredictionRun { prediction_run_id: string; dataset_id: string; model_ids: string[]; observation_month: string | null; store_ids: string[] | null; job_id: string; status: JobStatus; prediction_dir: string | null; created_at: string; error_message: string | null; }
export interface FilteredPredictionSummary { filters: { model_id: string | null; site_no: string | null; mc: string | null }; prediction_total: number; predicted_nonzero_book_count: number; mc_counts: Record<string, number>; predicted_20_plus_book_count: number; store_count: number; item_count: number; row_count: number; }
export interface PredictionSummary { prediction_run_id: string; dataset_id: string; observation_month: string; target_month: string | null; model_ids: string[]; model_summaries: Record<string, { prediction_total: number; predicted_nonzero_book_count: number; mc_counts: Record<string, number>; predicted_20_plus_book_count: number; store_count: number; item_count: number; row_count: number }>; filtered_summary: FilteredPredictionSummary; }
export interface NotificationRecord { notification_id: string; type: 'TEST' | 'PREDICTION_SUCCESS' | 'PREDICTION_FAILED' | 'EXPORT_SUCCESS'; target_email: string; status: 'SUCCESS' | 'FAILED'; related_run_id: string | null; created_at: string; error_message: string | null; }
export interface NotificationSettings { target_email: string; enabled_types: string[]; updated_at: string | null; }
export interface SmtpStatus { status: string; configured: boolean; }
export interface UploadProgress { percent?: number; loaded: number; total?: number; computable: boolean; }
export interface DifficultBookRow { site_no: string; item_id: string; book_name: string | null; observation_month: string | null; target_month: string | null; actual_qty: number; pred_qty: number; abs_error: number; stable_error: number; pred_mc: string; model_id: string; difficulty_level: string; }
export interface FieldSchema { field: string; label: string; description: string; }
export interface DifficultyRules { metric: string; stable_error_denominator: string; levels: Record<string, { min_abs_error: number; min_stable_error: number }>; note: string; }
export interface DifficultBooksPage { items: DifficultBookRow[]; total: number; page: number; page_size: number; field_schema: FieldSchema[]; difficulty_rules: DifficultyRules; }
export interface DifficultBooksSummary { difficult_book_count: number; difficult_ratio: number; store_count: number; avg_abs_error: number | null; hard_count: number; field_schema: FieldSchema[]; difficulty_rules: DifficultyRules; }
export interface HistoricalPoint { month: string; actual_qty: number | null; pred_qty: number | null; is_prediction: boolean; }
export interface HistoricalSeries { prediction_run_id: string; model_id: string; site_no: string | null; item_id: string | null; items: HistoricalPoint[]; }
export interface DataMonth { month: string; year: string; status: string; store_count: number; item_count: number; row_count: number; source_dataset_ids: string[]; }
export interface DataMonthCatalog { items: DataMonth[]; summary: { month_count: number; store_count: number; item_count: number; date_range: { start: string; end: string } | null }; lineage: Record<string, unknown>; }
export interface PredictionReadiness { status: 'READY' | 'NOT_READY'; target_month: string; observation_month: string; required_months?: string[]; missing_months?: string[]; affected_features?: string[]; contracts?: Record<string, unknown>[]; lineage?: Record<string, unknown>; detail?: string; }

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
export const getUploads = () => request<{ items: UploadRecord[] }>('/api/uploads');
export const getDatasets = () => request<{ items: Dataset[] }>('/api/datasets');
export const getDataMonths = () => request<DataMonthCatalog>('/api/datasets/months');
export const getDataset = (id: string) => request<Dataset>(`/api/datasets/${id}`);
export async function uploadFile(file: File) { const form = new FormData(); form.append('file', file); return request<UploadResult>('/api/uploads', { method: 'POST', body: form }); }
export function uploadFileWithProgress(file: File, onProgress: (progress: UploadProgress) => void): Promise<UploadResult> {
  if (!apiBaseUrl()) return Promise.reject(new Error('VITE_API_BASE_URL is required'));
  const form = new FormData();
  form.append('file', file);
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${apiBaseUrl()}/api/uploads`);
    xhr.upload.onprogress = event => {
      onProgress({
        computable: event.lengthComputable,
        loaded: event.loaded,
        total: event.lengthComputable ? event.total : undefined,
        percent: event.lengthComputable && event.total ? Math.round((event.loaded / event.total) * 100) : undefined,
      });
    };
    xhr.onload = () => {
      const body = xhr.responseText ? JSON.parse(xhr.responseText) : {};
      if (xhr.status >= 200 && xhr.status < 300) resolve(body as UploadResult);
      else reject(new Error(body.detail || `请求失败（${xhr.status}）`));
    };
    xhr.onerror = () => reject(new Error('暂无法连接预测服务'));
    xhr.send(form);
  });
}
export const processDataset = (upload_id: string, dataset_name: string, options: { processMode?: 'append' | 'create'; targetDatasetId?: string } = {}) => request<{ job_id: string; dataset_id: string }>('/api/datasets/process', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ upload_id, dataset_name, process_mode: options.processMode ?? 'create', target_dataset_id: options.targetDatasetId }) });
export const createPrediction = (payload: { dataset_id?: string; target_month?: string; model_ids: string[]; observation_month?: string; store_ids?: string[] }) => request<{ prediction_run_id: string; job_id: string; status: JobStatus }>('/api/predictions', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
export const checkPredictionReadiness = (payload: { target_month: string; model_ids: string[] }) => request<PredictionReadiness>('/api/predictions/readiness', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
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
export interface ResultFilters { modelId?: string; siteNo?: string; mc?: string; keyword?: string; page?: number; pageSize?: number; sortBy?: 'pred_qty_int' | 'p_sale' | 'item_id'; sortOrder?: 'asc' | 'desc'; }
export interface PagedResults { items: Record<string, unknown>[]; total: number; page: number; page_size: number; kind: string; sort?: { sort_by: string; sort_order: string }; }
export const getPredictionResults = (runId: string, kind: 'predictions' | 'top_books' | 'store_summary', filters: ResultFilters = {}) => {
  const params = new URLSearchParams({ kind });
  if (filters.modelId) params.set('model_id', filters.modelId);
  if (filters.siteNo) params.set('site_no', filters.siteNo);
  if (filters.mc) params.set('mc', filters.mc);
  if (filters.page) params.set('page', String(filters.page));
  if (filters.pageSize) params.set('page_size', String(filters.pageSize));
  if (filters.sortBy) params.set('sort_by', filters.sortBy);
  if (filters.sortOrder) params.set('sort_order', filters.sortOrder);
  return request<PagedResults>(`/api/predictions/${runId}/results?${params.toString()}`);
};
export const getStorePredictionSummary = (runId: string, siteNo: string, modelId: string) => request<{ prediction_total: number; pred_nonzero_count: number; pred_mc3_count: number; pred_mc4_count: number; item_count: number }>(`/api/predictions/${runId}/store/${encodeURIComponent(siteNo)}/summary?model_id=${encodeURIComponent(modelId)}`);
export const getHistoricalSeries = (runId: string, filters: { modelId?: string; siteNo?: string; itemId?: string } = {}) => {
  const params = new URLSearchParams();
  if (filters.modelId) params.set('model_id', filters.modelId);
  if (filters.siteNo) params.set('site_no', filters.siteNo);
  if (filters.itemId) params.set('item_id', filters.itemId);
  const query = params.toString();
  return request<HistoricalSeries>(`/api/predictions/${runId}/historical-series${query ? `?${query}` : ''}`);
};
export interface DifficultBookFilters { modelId?: string; siteNo?: string; difficultyLevel?: string; page?: number; pageSize?: number; }
const difficultBookParams = (filters: DifficultBookFilters = {}) => {
  const params = new URLSearchParams();
  if (filters.modelId) params.set('model_id', filters.modelId);
  if (filters.siteNo) params.set('site_no', filters.siteNo);
  if (filters.difficultyLevel) params.set('difficulty_level', filters.difficultyLevel);
  if (filters.page) params.set('page', String(filters.page));
  if (filters.pageSize) params.set('page_size', String(filters.pageSize));
  return params;
};
export const getDifficultBooks = (runId: string, filters: DifficultBookFilters = {}) => {
  const query = difficultBookParams(filters).toString();
  return request<DifficultBooksPage>(`/api/predictions/${runId}/difficult-books${query ? `?${query}` : ''}`);
};
export const getDifficultBooksSummary = (runId: string, filters: DifficultBookFilters = {}) => {
  const params = difficultBookParams(filters);
  params.delete('page');
  params.delete('page_size');
  const query = params.toString();
  return request<DifficultBooksSummary>(`/api/predictions/${runId}/difficult-books/summary${query ? `?${query}` : ''}`);
};
export const predictionDownloadUrl = (runId: string) => `${apiBaseUrl()}/api/predictions/${runId}/download-parquet`;
export const difficultBooksExportUrl = (runId: string, filters: DifficultBookFilters = {}) => {
  const params = difficultBookParams(filters);
  params.delete('page');
  params.delete('page_size');
  const query = params.toString();
  return `${apiBaseUrl()}/api/predictions/${runId}/difficult-books/export${query ? `?${query}` : ''}`;
};
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
export const getNotificationSettings = () => request<NotificationSettings>('/api/notifications/settings');
export const saveNotificationSettings = (payload: { target_email: string; enabled_types: string[] }) => request<NotificationSettings>('/api/notifications/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
export const getSmtpStatus = () => request<SmtpStatus>('/api/notifications/smtp-status');
export const getNotifications = () => request<{ items: NotificationRecord[] }>('/api/notifications');
export const sendPredictionNotification = (runId: string, payload: { target_email: string; model_id?: string; include_excel_link?: boolean; notification_type?: 'PREDICTION_SUCCESS' | 'EXPORT_SUCCESS'; site_no?: string; mc?: string }) => request<NotificationRecord>(`/api/predictions/${runId}/notify`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
