import { describe, expect, it, vi } from 'vitest';
import { apiBaseUrl, checkPredictionReadiness, createPrediction, difficultBooksExportUrl, getDataMonths, getDifficultBooks, getDifficultBooksSummary, getHealth, getPredictionResults, getPredictionSummary, getUploads, predictionExcelUrl, processDataset, sendTestEmail, uploadFileWithProgress } from './api';

describe('API client', () => {
  it('uses VITE_API_BASE_URL for health requests', async () => {
    vi.stubEnv('VITE_API_BASE_URL', 'http://localhost:9123');
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ status: 'ok' })));
    vi.stubGlobal('fetch', fetchMock);

    await expect(getHealth()).resolves.toEqual({ status: 'ok' });
    expect(apiBaseUrl()).toBe('http://localhost:9123');
    expect(fetchMock).toHaveBeenCalledWith('http://localhost:9123/api/health', expect.any(Object));
  });

  it('passes prediction filters and pagination to the backend', async () => {
    vi.stubEnv('VITE_API_BASE_URL', 'http://localhost:9123');
    const fetchMock = vi.fn().mockImplementation(() => Promise.resolve(new Response(JSON.stringify({ items: [], total: 0 }))));
    vi.stubGlobal('fetch', fetchMock);

    await getPredictionResults('run-1', 'predictions', { modelId: 'E2', siteNo: '5000', mc: 'MC4', page: 2, pageSize: 25, sortBy: 'p_sale', sortOrder: 'asc' });

    expect(fetchMock).toHaveBeenCalledWith('http://localhost:9123/api/predictions/run-1/results?kind=predictions&model_id=E2&site_no=5000&mc=MC4&page=2&page_size=25&sort_by=p_sale&sort_order=asc', expect.any(Object));
  });

  it('passes current filters to filtered summary and Excel export URLs', async () => {
    vi.stubEnv('VITE_API_BASE_URL', 'http://localhost:9123');
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ filtered_summary: {} })));
    vi.stubGlobal('fetch', fetchMock);

    await getPredictionSummary('run-1', { modelId: 'E2', siteNo: '8000', mc: 'MC4' });

    expect(fetchMock).toHaveBeenCalledWith('http://localhost:9123/api/predictions/run-1/summary?model_id=E2&site_no=8000&mc=MC4', expect.any(Object));
    expect(predictionExcelUrl('run-1', { modelId: 'E2', siteNo: '8000', mc: 'MC4', includePredictions: true, topN: 100 }))
      .toBe('http://localhost:9123/api/predictions/run-1/export-excel?model_id=E2&site_no=8000&mc=MC4&include_predictions=true&top_n=100');
  });

  it('posts a test email recipient to the notification endpoint', async () => {
    vi.stubEnv('VITE_API_BASE_URL', 'http://localhost:9123');
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ status: 'SUCCESS' })));
    vi.stubGlobal('fetch', fetchMock);

    await sendTestEmail('business@example.test');

    expect(fetchMock).toHaveBeenCalledWith('http://localhost:9123/api/notifications/test-email', expect.objectContaining({ method: 'POST' }));
  });

  it('loads upload records and sends append processing intent', async () => {
    vi.stubEnv('VITE_API_BASE_URL', 'http://localhost:9123');
    const fetchMock = vi.fn().mockImplementation(() => Promise.resolve(new Response(JSON.stringify({ items: [] }))));
    vi.stubGlobal('fetch', fetchMock);

    await getUploads();
    await processDataset('upload-1', '文轩销售主数据', { processMode: 'append', targetDatasetId: 'dataset-1' });

    expect(fetchMock).toHaveBeenNthCalledWith(1, 'http://localhost:9123/api/uploads', expect.any(Object));
    expect(fetchMock).toHaveBeenNthCalledWith(2, 'http://localhost:9123/api/datasets/process', expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({ upload_id: 'upload-1', dataset_name: '文轩销售主数据', process_mode: 'append', target_dataset_id: 'dataset-1' }),
    }));
  });

  it('uses standard history endpoints for future prediction flow', async () => {
    vi.stubEnv('VITE_API_BASE_URL', 'http://localhost:9123');
    const fetchMock = vi.fn().mockImplementation(() => Promise.resolve(new Response(JSON.stringify({ status: 'READY', items: [] }))));
    vi.stubGlobal('fetch', fetchMock);

    await getDataMonths();
    await checkPredictionReadiness({ target_month: '2026-07', model_ids: ['E0', 'E2', 'E3'] });
    await createPrediction({ target_month: '2026-07', model_ids: ['E2'] });

    expect(fetchMock).toHaveBeenNthCalledWith(1, 'http://localhost:9123/api/datasets/months', expect.any(Object));
    expect(fetchMock).toHaveBeenNthCalledWith(2, 'http://localhost:9123/api/predictions/readiness', expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({ target_month: '2026-07', model_ids: ['E0', 'E2', 'E3'] }),
    }));
    expect(fetchMock).toHaveBeenNthCalledWith(3, 'http://localhost:9123/api/predictions', expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({ target_month: '2026-07', model_ids: ['E2'] }),
    }));
  });

  it('passes difficult book filters to list, summary, and export endpoints', async () => {
    vi.stubEnv('VITE_API_BASE_URL', 'http://localhost:9123');
    const fetchMock = vi.fn().mockImplementation(() => Promise.resolve(new Response(JSON.stringify({ items: [], total: 0 }))));
    vi.stubGlobal('fetch', fetchMock);

    await getDifficultBooks('run-1', { modelId: 'E2', siteNo: '8000', difficultyLevel: '困难', page: 2, pageSize: 25 });
    await getDifficultBooksSummary('run-1', { modelId: 'E2', siteNo: '8000', difficultyLevel: '困难' });

    expect(fetchMock).toHaveBeenNthCalledWith(1, 'http://localhost:9123/api/predictions/run-1/difficult-books?model_id=E2&site_no=8000&difficulty_level=%E5%9B%B0%E9%9A%BE&page=2&page_size=25', expect.any(Object));
    expect(fetchMock).toHaveBeenNthCalledWith(2, 'http://localhost:9123/api/predictions/run-1/difficult-books/summary?model_id=E2&site_no=8000&difficulty_level=%E5%9B%B0%E9%9A%BE', expect.any(Object));
    expect(difficultBooksExportUrl('run-1', { modelId: 'E2', siteNo: '8000', difficultyLevel: '困难' }))
      .toBe('http://localhost:9123/api/predictions/run-1/difficult-books/export?model_id=E2&site_no=8000&difficulty_level=%E5%9B%B0%E9%9A%BE');
  });

  it('uses XHR upload progress events for real upload feedback', async () => {
    vi.stubEnv('VITE_API_BASE_URL', 'http://localhost:9123');
    const progressEvents: number[] = [];
    class FakeXhr {
      upload = {} as { onprogress?: (event: ProgressEvent) => void };
      status = 201;
      responseText = JSON.stringify({ upload_id: 'upload-1', job_id: 'job-1', filename: 'sales.csv', stored_path: 'stored', csv_files: [], headers: [], field_mapping: {}, mapping_status: 'READY', created_at: 'now' });
      onload: (() => void) | null = null;
      onerror: (() => void) | null = null;
      open = vi.fn();
      send = vi.fn(() => {
        this.upload.onprogress?.({ lengthComputable: true, loaded: 50, total: 100 } as ProgressEvent);
        this.onload?.();
      });
    }
    vi.stubGlobal('XMLHttpRequest', FakeXhr);

    await expect(uploadFileWithProgress(new File(['id'], 'sales.csv'), progress => progressEvents.push(progress.percent ?? -1))).resolves.toMatchObject({ upload_id: 'upload-1' });
    expect(progressEvents).toEqual([50]);
  });
});
