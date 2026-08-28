import { describe, expect, it, vi } from 'vitest';
import { apiBaseUrl, getHealth, getPredictionResults, sendTestEmail } from './api';

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
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ items: [], total: 0 })));
    vi.stubGlobal('fetch', fetchMock);

    await getPredictionResults('run-1', 'predictions', { modelId: 'E2', siteNo: '5000', mc: 'MC4', page: 2, pageSize: 25 });

    expect(fetchMock).toHaveBeenCalledWith('http://localhost:9123/api/predictions/run-1/results?kind=predictions&model_id=E2&site_no=5000&mc=MC4&page=2&page_size=25', expect.any(Object));
  });

  it('posts a test email recipient to the notification endpoint', async () => {
    vi.stubEnv('VITE_API_BASE_URL', 'http://localhost:9123');
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ status: 'SUCCESS' })));
    vi.stubGlobal('fetch', fetchMock);

    await sendTestEmail('business@example.test');

    expect(fetchMock).toHaveBeenCalledWith('http://localhost:9123/api/notifications/test-email', expect.objectContaining({ method: 'POST' }));
  });
});
