import { useEffect, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { Alert, Button, Card, Checkbox, Col, Collapse, Descriptions, Drawer, Empty, Form, Input, Layout, Menu, Modal, Progress, Radio, Row, Select, Space, Spin, Statistic, Table, Tabs, Tag, Typography, Upload, message } from 'antd';
import type { ColumnsType, TablePaginationConfig } from 'antd/es/table';
import * as echarts from 'echarts';
import type { EChartsOption } from 'echarts';
import { checkPredictionReadiness, createPrediction, DataMonthCatalog, Dataset, DifficultBooksPage, DifficultBooksSummary, difficultBooksExportUrl, getDataMonths, getDatasets, getDifficultBooks, getDifficultBooksSummary, getHealth, getHistoricalSeries, getJobs, getNotificationSettings, getNotifications, getPredictionResults, getPredictions, getPredictionSummary, getSmtpStatus, getStorePredictionSummary, getUploads, HistoricalSeries, Job, NotificationRecord, NotificationSettings, PagedResults, PredictionReadiness, PredictionRun, PredictionSummary, predictionExcelUrl, processDataset, saveNotificationSettings, sendPredictionNotification, sendTestEmail, SmtpStatus, uploadFileWithProgress, UploadProgress, UploadRecord, UploadResult } from './lib/api';

const { Header, Sider, Content } = Layout;
const { Text, Title } = Typography;

type View = 'overview' | 'data' | 'predict' | 'analysis' | 'difficult' | 'models' | 'tasks' | 'notify';
type LoadState = 'idle' | 'loading' | 'success' | 'error';
type SortBy = 'pred_qty_int' | 'p_sale' | 'item_id';
type SortOrder = 'asc' | 'desc';

const queryCache = new Map<string, unknown>();
const cached = async <T,>(key: string, loader: () => Promise<T>, force = false): Promise<T> => {
  if (!force && queryCache.has(key)) return queryCache.get(key) as T;
  const value = await loader();
  queryCache.set(key, value);
  return value;
};
const clearCache = () => queryCache.clear();

const mcLabels: Record<string, string> = { MC0: '无动销（MC0）', MC1: '低动销（MC1）', MC2: '一般（MC2）', MC3: '较高动销（MC3）', MC4: '高动销（MC4）' };
const modelNames: Record<string, string> = { E0: '基础 Two-stage（E0）', E2: '跨门店增强 Two-stage（E2）', E3: '差分 + 跨门店增强（E3）' };
const runnableModels = ['E0', 'E2', 'E3'];
const notificationTypeOptions = [
  { label: '预测完成', value: 'PREDICTION_SUCCESS' },
  { label: '预测失败', value: 'PREDICTION_FAILED' },
  { label: '数据处理完成', value: 'DATA_PROCESSING_SUCCESS' },
  { label: '数据处理失败', value: 'DATA_PROCESSING_FAILED' },
  { label: '导出完成', value: 'EXPORT_SUCCESS' },
];
const modelDetails = [
  { category: '统计预测基线', name: '历史均值', tech: 'Historical Average', status: '实验对照', formal: false, flow: '按历史销量均值形成基线预测。', why: '用于判断复杂模型是否真正带来改进。', outputs: '预测销量。' },
  { category: '统计预测基线', name: '加权移动平均', tech: 'Weighted Moving Average', status: '实验对照', formal: false, flow: '近期月份权重更高，形成平滑预测。', why: '适合做简单、可解释的时间序列基线。', outputs: '预测销量。' },
  { category: '机器学习', name: '随机森林', tech: 'Random Forest', status: '历史实验模型', formal: false, flow: '基于构造特征进行树模型回归或分类。', why: '作为非线性机器学习对照。', outputs: '预测销量或分类结果。' },
  { category: '机器学习', name: 'LightGBM', tech: 'LightGBM', status: '历史实验模型', formal: false, flow: '使用梯度提升树学习门店、图书、月份等特征。', why: '适合表格特征和大规模样本。', outputs: '预测销量。' },
  { category: '神经网络', name: 'MLP', tech: 'MLP', status: '历史实验模型', formal: false, flow: '用多层感知机拟合特征到销量的映射。', why: '作为神经网络对照实验。', outputs: '预测销量。' },
  { category: '长尾 / 两阶段', name: '基础 Two-stage（E0）', tech: 'Two-stage', status: '可用于当前正式预测', formal: true, flow: '第一阶段预测是否发生销量，输出 p_sale；第二阶段预测动销条件下销量，输出 conditional_qty；最终使用 p_sale × conditional_qty。', why: '图书销量大量为 0，且存在长尾和间歇性需求，两阶段结构更贴近业务分布。', outputs: 'p_sale、conditional_qty、预测销量、动销等级。' },
  { category: '增强模型', name: '跨门店增强 Two-stage（E2）', tech: 'Cross-store Two-stage', status: '可用于当前正式预测', formal: true, flow: '在 Two-stage 基础上加入跨门店图书需求摘要特征。', why: '同一本书在不同门店之间存在需求关联，可帮助识别跨店动销信号。', outputs: 'p_sale、conditional_qty、预测销量、动销等级。' },
  { category: '增强模型', name: '差分 + 跨门店增强（E3）', tech: 'Diff + Cross-store', status: '可用于当前正式预测', formal: true, flow: '在跨门店增强基础上加入差分变化特征。', why: '帮助捕捉近期销量变化趋势。', outputs: 'p_sale、conditional_qty、预测销量、动销等级。' },
];

const formatNumber = (value?: number | null) => value === undefined || value === null ? '暂无' : value.toLocaleString('zh-CN');
const fixed = (value: unknown) => typeof value === 'number' ? value.toFixed(4) : '—';
const percent = (value?: number | null) => value === undefined || value === null ? '暂无' : `${(value * 100).toFixed(2)}%`;
const fileSize = (bytes?: number | null) => bytes === undefined || bytes === null ? '未知大小' : bytes < 1024 * 1024 ? `${(bytes / 1024).toFixed(1)} KB` : `${(bytes / 1024 / 1024).toFixed(1)} MB`;
const displayTime = (value?: string | null) => value ? new Intl.DateTimeFormat('zh-CN', { year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false }).format(new Date(value)).replace(/\//g, '-') : '未完成';
const targetMonth = (observation?: string | null, explicit?: string | null) => explicit || nextMonth(observation);
const nextMonth = (month?: string | null) => {
  if (!month) return '暂无';
  const [year, rawMonth] = month.split('-').map(Number);
  const next = rawMonth === 12 ? { y: year + 1, m: 1 } : { y: year, m: rawMonth + 1 };
  return `${next.y}-${String(next.m).padStart(2, '0')}`;
};
const statusTag = (value: string) => <Tag color={{ SUCCESS: 'success', FAILED: 'error', RUNNING: 'processing', READY: 'success', QUEUED: 'default' }[value] || 'default'}>{({ SUCCESS: '成功', FAILED: '失败', RUNNING: '运行中', READY: '就绪', QUEUED: '排队中' } as Record<string, string>)[value] || value}</Tag>;
const runLabel = (run: PredictionRun, summary?: PredictionSummary) => `预测 ${targetMonth(summary?.observation_month ?? run.observation_month, summary?.target_month)} · ${run.model_ids.join('/')} · ${displayTime(run.created_at).slice(5, 16)}`;

function App() {
  const [view, setView] = useState<View>('overview');
  const [health, setHealth] = useState<LoadState>('loading');
  const [jobs, setJobs] = useState<Job[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [runs, setRuns] = useState<PredictionRun[]>([]);
  const refresh = async (force = false) => {
    if (force) clearCache();
    setHealth('loading');
    try {
      const [h, j, d, r] = await Promise.all([
        cached('health', getHealth, force), cached('jobs', getJobs, force), cached('datasets', getDatasets, force), cached('predictions', getPredictions, force),
      ]);
      setHealth(h.status === 'ok' ? 'success' : 'error');
      setJobs(j.items); setDatasets(d.items); setRuns(r.items);
    } catch {
      setHealth('error');
    }
  };
  useEffect(() => { void refresh(); }, []);
  const latestSuccessRun = runs.find(run => run.status === 'SUCCESS');
  const pages: Record<View, ReactNode> = {
    overview: <OverviewPage latestRun={latestSuccessRun} datasets={datasets} onNavigate={setView} />,
    data: <DataAccessPage datasets={datasets} refresh={() => refresh(true)} />,
    predict: <PredictionPage datasets={datasets} refresh={() => refresh(true)} onNavigate={setView} />,
    analysis: <AnalysisPage runs={runs} />,
    difficult: <DifficultBooksPageView runs={runs} />,
    models: <ModelsPage />,
    tasks: <TasksPage jobs={jobs} refresh={() => refresh(true)} />,
    notify: <NotificationsPage />,
  };
  return <Layout className="app-shell"><Sider width={232} theme="dark" className="sidebar"><div className="brand">文轩集团<br /><span>图书销量预测系统</span></div><Menu theme="dark" mode="inline" selectedKeys={[view]} onClick={({ key }) => setView(key as View)} items={[['overview', '系统总览'], ['data', '数据接入'], ['predict', '销量预测'], ['analysis', '预测分析'], ['difficult', '难预测图书'], ['models', '模型说明'], ['tasks', '任务管理'], ['notify', '通知设置']].map(([key, label]) => ({ key, label }))} /></Sider><Layout><Header className="topbar"><Space><Text strong>输入数据 → 数据处理 → 模型预测 → 类别提取 → 结果展示</Text>{statusTag(health === 'success' ? 'SUCCESS' : health === 'loading' ? 'RUNNING' : 'FAILED')}<Button onClick={() => void refresh(true)}>刷新</Button></Space></Header><Content className="content">{pages[view]}</Content></Layout></Layout>;
}

export default App;

function PageHead({ title, actions }: { title: string; actions?: ReactNode }) {
  return <div className="page-head"><Title level={2}>{title}</Title><div className="head-actions">{actions}</div></div>;
}

function MetricCard({ title, value, suffix }: { title: string; value: ReactNode; suffix?: string }) {
  return <Card className="metric-card"><div className="metric-label">{title}</div><div className="metric-value">{value}</div>{suffix && <div className="metric-foot">{suffix}</div>}</Card>;
}

function Chart({ option, height = 260 }: { option: EChartsOption; height?: number }) {
  const ref = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (!ref.current) return;
    const chart = echarts.init(ref.current);
    chart.setOption(option);
    const resize = () => chart.resize();
    window.addEventListener('resize', resize);
    return () => { window.removeEventListener('resize', resize); chart.dispose(); };
  }, [option]);
  return <div ref={ref} style={{ height }} />;
}

function useCachedData<T>(key: string | undefined, loader: () => Promise<T>, deps: unknown[]) {
  const [data, setData] = useState<T>();
  const [state, setState] = useState<LoadState>('idle');
  useEffect(() => {
    if (!key) return;
    let cancelled = false;
    setState(queryCache.has(key) ? 'success' : 'loading');
    cached(key, loader).then(value => { if (!cancelled) { setData(value); setState('success'); } }).catch(() => { if (!cancelled) setState('error'); });
    return () => { cancelled = true; };
  }, deps);
  return { data, state };
}

function OverviewPage({ latestRun, datasets, onNavigate }: { latestRun?: PredictionRun; datasets: Dataset[]; onNavigate: (view: View) => void }) {
  const { data: summary, state } = useCachedData(latestRun ? `summary:${latestRun.prediction_run_id}` : undefined, () => getPredictionSummary(latestRun!.prediction_run_id), [latestRun?.prediction_run_id]);
  const modelId = summary?.model_ids[0];
  const { data: difficult } = useCachedData(latestRun && modelId ? `difficult-summary:${latestRun.prediction_run_id}:${modelId}` : undefined, () => getDifficultBooksSummary(latestRun!.prediction_run_id, { modelId }), [latestRun?.prediction_run_id, modelId]);
  const current = modelId ? summary?.model_summaries[modelId] : undefined;
  return <>
    <PageHead title="系统总览" actions={<Button type="primary" onClick={() => onNavigate('predict')}>发起新预测</Button>} />
    <Card className="hero-card"><div><h3>未来一个月图书销量预测与动销分析</h3><p>首页默认展示最近一次成功完成的正式预测任务，不展示失败或运行中的任务。</p></div><Space><Button type="primary" onClick={() => onNavigate('analysis')}>查看最新预测结果</Button><Button onClick={() => onNavigate('difficult')}>查看难预测图书</Button></Space></Card>
    {!latestRun && <Empty className="page-empty" description="暂无预测结果，请先完成数据接入并发起预测。" />}
    {latestRun && state === 'loading' && <Spin />}
    {latestRun && state === 'error' && <Alert type="error" showIcon message="加载失败，请重试" />}
    {latestRun && summary && <Card className="source-card"><Descriptions size="small" column={4}><Descriptions.Item label="预测目标月份">{summary.target_month}</Descriptions.Item><Descriptions.Item label="模型">{modelNames[modelId ?? ''] ?? modelId}</Descriptions.Item><Descriptions.Item label="数据截止月份">{summary.observation_month}</Descriptions.Item><Descriptions.Item label="任务创建时间">{displayTime(latestRun.created_at)}</Descriptions.Item><Descriptions.Item label="技术任务 ID">{latestRun.prediction_run_id}</Descriptions.Item></Descriptions></Card>}
    {current && <Row gutter={[12, 12]}><Col span={4}><MetricCard title="预测目标月份" value={summary?.target_month ?? '暂无'} /></Col><Col span={4}><MetricCard title="预测总销量" value={formatNumber(current.prediction_total)} /></Col><Col span={4}><MetricCard title="预计动销图书数" value={formatNumber(current.predicted_nonzero_book_count)} /></Col><Col span={4}><MetricCard title="高动销图书数" value={formatNumber(current.mc_counts.MC4 ?? 0)} /></Col><Col span={4}><MetricCard title="覆盖门店数" value={formatNumber(current.store_count)} /></Col><Col span={4}><MetricCard title="难预测图书数" value={difficult?.difficult_book_count ? formatNumber(difficult.difficult_book_count) : '暂无验证结果'} /></Col></Row>}
    <Row gutter={[16, 16]} className="section-grid"><Col span={12}><PredictionTrend run={latestRun} modelId={modelId} /></Col><Col span={12}><McChart summary={current} /></Col><Col span={12}><TopBooksPanel run={latestRun} modelId={modelId} /></Col><Col span={12}><StorePanel run={latestRun} modelId={modelId} /></Col></Row>
    {datasets.length === 0 && <Alert type="info" showIcon message="暂无已接入数据集" description="请先在“数据接入”上传 CSV 或 ZIP，并完成数据处理。" />}
  </>;
}

function PredictionTrend({ run, modelId, siteNo, itemId }: { run?: PredictionRun; modelId?: string; siteNo?: string; itemId?: string }) {
  const key = run && modelId ? `series:${run.prediction_run_id}:${modelId}:${siteNo ?? 'all'}:${itemId ?? 'all'}` : undefined;
  const { data: series, state } = useCachedData<HistoricalSeries>(key, () => getHistoricalSeries(run!.prediction_run_id, { modelId, siteNo, itemId }), [key]);
  if (!run || !modelId) return <Card title="历史销量 / 预测销量趋势"><Empty description="暂无预测结果" /></Card>;
  if (state === 'error') return <Card title="历史销量 / 预测销量趋势"><Empty description="暂无历史趋势接口数据" /></Card>;
  const points = series?.items ?? [];
  if (!points.length) return <Card title="历史销量 / 预测销量趋势"><Spin /></Card>;
  return <Card title="历史销量 / 预测销量趋势"><Chart option={{ tooltip: { trigger: 'axis' }, legend: { data: ['历史实际销量', '下一期预测销量'] }, xAxis: { type: 'category', data: points.map(p => p.month) }, yAxis: { type: 'value', name: '销量' }, series: [{ name: '历史实际销量', type: 'line', data: points.map(p => p.actual_qty), smooth: true }, { name: '下一期预测销量', type: 'line', data: points.map(p => p.pred_qty), symbolSize: 10, lineStyle: { type: 'dashed' } }] }} /></Card>;
}

function McChart({ summary }: { summary?: PredictionSummary['filtered_summary'] | PredictionSummary['model_summaries'][string] }) {
  const counts = summary?.mc_counts;
  if (!counts) return <Card title="MC 动销等级结构"><Empty description="暂无数据" /></Card>;
  return <Card title="MC 动销等级结构"><Chart option={{ tooltip: { trigger: 'item' }, xAxis: { type: 'category', data: Object.keys(mcLabels).map(key => mcLabels[key]) }, yAxis: { type: 'value', name: '图书数' }, series: [{ type: 'bar', data: Object.keys(mcLabels).map(key => counts[key] ?? 0), itemStyle: { color: '#2563eb' } }] }} /></Card>;
}

function TopBooksPanel({ run, modelId, siteNo }: { run?: PredictionRun; modelId?: string; siteNo?: string }) {
  const key = run && modelId ? `top:${run.prediction_run_id}:${modelId}:${siteNo ?? 'all'}` : undefined;
  const { data } = useCachedData<PagedResults>(key, () => getPredictionResults(run!.prediction_run_id, 'top_books', { modelId, siteNo, pageSize: 8 }), [key]);
  const rows = data?.items ?? [];
  return <Card title="预测销量 Top 图书">{rows.length ? <Table size="small" rowKey={row => `${row.site_no}-${row.item_id}-${row.model_id}`} pagination={false} dataSource={rows} columns={[{ title: '门店', dataIndex: 'site_no' }, { title: '图书编号', dataIndex: 'item_id' }, { title: '图书名称', dataIndex: 'book_name' }, { title: '预测销量', dataIndex: 'pred_qty_int' }, { title: '动销等级', dataIndex: 'pred_mc', render: (v: string) => mcLabels[v] || v }]} /> : <Empty description="暂无 Top 图书数据" />}</Card>;
}

function StorePanel({ run, modelId }: { run?: PredictionRun; modelId?: string }) {
  const key = run && modelId ? `stores:${run.prediction_run_id}:${modelId}` : undefined;
  const { data } = useCachedData<PagedResults>(key, () => getPredictionResults(run!.prediction_run_id, 'store_summary', { modelId, pageSize: 8 }), [key]);
  const rows = data?.items ?? [];
  return <Card title="重点门店">{rows.length ? <Table size="small" rowKey={row => `${row.site_no}-${row.model_id}`} pagination={false} dataSource={rows} columns={[{ title: '门店编号', dataIndex: 'site_no' }, { title: '预计总销量', dataIndex: 'pred_total' }, { title: '预计动销图书', dataIndex: 'pred_nonzero_count' }, { title: '高动销图书', dataIndex: 'pred_20_plus_count' }]} /> : <Empty description="暂无门店汇总数据" />}</Card>;
}

function LegacyDataAccessPage({ datasets, refresh }: { datasets: Dataset[]; refresh: () => Promise<void> }) {
  const { data: uploads, state } = useCachedData<{ items: UploadRecord[] }>('uploads', getUploads, []);
  const [upload, setUpload] = useState<UploadResult>(); const [progress, setProgress] = useState<UploadProgress>(); const [uploadState, setUploadState] = useState<'待上传' | '上传中' | '上传完成' | '数据处理中' | '处理完成' | '失败'>('待上传'); const [file, setFile] = useState<File>(); const [form] = Form.useForm(); const [busy, setBusy] = useState(false); const [mode, setMode] = useState<'append' | 'create'>(datasets.length ? 'append' : 'create');
  const customRequest = ({ file: selectedFile, onSuccess, onError }: any) => {
    const realFile = selectedFile as File; setFile(realFile); setUpload(undefined); setProgress(undefined); setUploadState('上传中');
    uploadFileWithProgress(realFile, setProgress).then(result => { setUpload(result); setUploadState('上传完成'); clearCache(); message.success('文件上传完成，请选择处理方式。'); onSuccess?.(result); }).catch(error => { setUploadState('失败'); message.error(String(error)); onError?.(error); });
  };
  const process = async () => {
    if (!upload) return;
    try { const values = await form.validateFields(); setBusy(true); setUploadState('数据处理中'); const datasetName = mode === 'append' ? datasets.find(d => d.dataset_id === values.target_dataset_id)?.dataset_name ?? values.name : values.name; await processDataset(upload.upload_id, datasetName, { processMode: mode, targetDatasetId: values.target_dataset_id }); await refresh(); setUploadState('处理完成'); message.success(mode === 'append' ? '数据处理完成。当前版本已预留追加入口，处理产物按新数据集登记。' : '数据处理完成，数据集已就绪。'); } catch (error) { setUploadState('失败'); message.error(String(error).includes('dataset_name') ? '请输入数据集名称' : String(error)); } finally { setBusy(false); }
  };
  return <><PageHead title="数据接入" actions={<Tag color="blue">支持 CSV / ZIP</Tag>} /><Row gutter={[16, 16]}><Col span={10}><Card title="新数据上传"><Upload.Dragger accept=".csv,.zip" maxCount={1} customRequest={customRequest}><p>点击或拖拽上传 CSV / ZIP</p><p className="muted">离开页面后不恢复浏览器文件框，但上传记录会持久显示。</p></Upload.Dragger><div className="upload-status"><Text strong>{file?.name ?? '尚未选择文件'}</Text><Text type="secondary">{fileSize(file?.size)}</Text>{statusTag(uploadState === '失败' ? 'FAILED' : uploadState === '处理完成' || uploadState === '上传完成' ? 'SUCCESS' : uploadState === '待上传' ? 'QUEUED' : 'RUNNING')}<Progress percent={progress?.percent} status={uploadState === '失败' ? 'exception' : undefined} showInfo={progress?.computable} /></div>{upload && <Form form={form} layout="vertical" className="inline-form"><Form.Item label="处理方式"><Radio.Group value={mode} onChange={e => setMode(e.target.value)}><Radio value="append" disabled={!datasets.length}>追加到已有数据集（推荐）</Radio><Radio value="create">创建新数据集</Radio></Radio.Group></Form.Item>{mode === 'append' ? <Form.Item name="target_dataset_id" label="选择逻辑数据集" rules={[{ required: true, message: '请选择要追加的数据集' }]}><Select options={datasets.map(d => ({ value: d.dataset_id, label: `${d.dataset_name}（${d.date_range.start} 至 ${d.date_range.end}）` }))} /></Form.Item> : <Form.Item name="name" label="新数据集名称" rules={[{ required: true, message: '请输入数据集名称' }]}><Input placeholder="例如：文轩销售主数据" /></Form.Item>}<Button type="primary" loading={busy} disabled={upload.mapping_status !== 'READY'} onClick={() => void process()}>开始处理</Button></Form>}</Card></Col><Col span={14}><Card title="上传记录">{state === 'loading' ? <Spin /> : <Table rowKey="upload_id" dataSource={uploads?.items ?? []} locale={{ emptyText: '暂无上传记录' }} columns={[{ title: '文件名', dataIndex: 'filename' }, { title: '文件大小', dataIndex: 'file_size_bytes', render: fileSize }, { title: '上传时间', dataIndex: 'created_at', render: displayTime }, { title: '上传状态', render: () => statusTag('SUCCESS') }, { title: '处理状态', dataIndex: 'processing_status' }, { title: '关联数据集', render: (_, r: UploadRecord) => r.related_dataset?.dataset_name ?? '未关联' }]} expandable={{ expandedRowRender: r => <Descriptions size="small" column={1}><Descriptions.Item label="上传 ID">{r.upload_id}</Descriptions.Item><Descriptions.Item label="存储路径">{r.stored_path}</Descriptions.Item><Descriptions.Item label="CSV 文件">{r.csv_files.join('；')}</Descriptions.Item></Descriptions> }} />}</Card></Col></Row><Card className="table-card" title="数据集列表"><Table rowKey="dataset_id" dataSource={datasets} columns={[{ title: '逻辑数据集', dataIndex: 'dataset_name' }, { title: '覆盖区间', render: (_, r: Dataset) => `${r.date_range.start} 至 ${r.date_range.end}` }, { title: '门店数', dataIndex: 'store_count' }, { title: '图书数', dataIndex: 'item_count' }, { title: '特征状态', render: (_, r: Dataset) => <Space>{r.has_active_store && <Tag>Active-Store</Tag>}{r.has_cross_store_features && <Tag>跨门店特征</Tag>}{r.has_diff_features && <Tag>差分特征</Tag>}</Space> }, { title: '数据状态', dataIndex: 'status', render: statusTag }]} /></Card></>;
}

function DataAccessPage({ datasets, refresh }: { datasets: Dataset[]; refresh: () => Promise<void> }) {
  return <><LegacyDataAccessPage datasets={datasets} refresh={refresh} /><DataMonthBrowser /></>;
}

function DataMonthBrowser() {
  const { data, state } = useCachedData<DataMonthCatalog>('data-months', getDataMonths, []);
  const years = useMemo(() => {
    const grouped = new Map<string, DataMonthCatalog['items']>();
    for (const item of data?.items ?? []) grouped.set(item.year, [...(grouped.get(item.year) ?? []), item]);
    return Array.from(grouped.entries()).sort(([a], [b]) => b.localeCompare(a));
  }, [data]);
  return <Card className="table-card" title="标准历史数据">{state === 'loading' ? <Spin /> : <><Descriptions size="small" column={4}><Descriptions.Item label="当前覆盖">{data?.summary.date_range ? `${data.summary.date_range.start} 至 ${data.summary.date_range.end}` : '暂无'}</Descriptions.Item><Descriptions.Item label="月份数">{formatNumber(data?.summary.month_count)}</Descriptions.Item><Descriptions.Item label="门店数">{formatNumber(data?.summary.store_count)}</Descriptions.Item><Descriptions.Item label="图书数">{formatNumber(data?.summary.item_count)}</Descriptions.Item></Descriptions><Collapse items={years.map(([year, items]) => ({ key: year, label: year, children: <Table rowKey="month" pagination={false} dataSource={items} columns={[{ title: '月份', dataIndex: 'month' }, { title: '状态', dataIndex: 'status', render: statusTag }, { title: '覆盖门店', dataIndex: 'store_count' }, { title: '覆盖图书', dataIndex: 'item_count' }, { title: '月度行数', dataIndex: 'row_count' }, { title: '来源数据集', render: (_, row) => row.source_dataset_ids.join('、') || '暂无' }]} /> }))} /></>}</Card>;
}

function LegacyPredictionPage({ datasets, refresh, onNavigate }: { datasets: Dataset[]; refresh: () => Promise<void>; onNavigate: (view: View) => void }) {
  const [form] = Form.useForm(); const [busy, setBusy] = useState(false); const [selectedDatasetId, setSelectedDatasetId] = useState<string>(); const ready = datasets.filter(d => d.status === 'READY'); const selected = ready.find(d => d.dataset_id === selectedDatasetId) ?? ready[0]; const observation = selected?.date_range.end; const target = nextMonth(observation);
  useEffect(() => { if (ready[0] && !selectedDatasetId) setSelectedDatasetId(ready[0].dataset_id); }, [ready, selectedDatasetId]);
  const submit = async () => { if (!selected) return; const values = await form.validateFields(); setBusy(true); try { await createPrediction({ dataset_id: selected.dataset_id, model_ids: values.model_ids, observation_month: observation }); await refresh(); message.success('预测完成，已生成结果。'); onNavigate('analysis'); } catch (error) { message.error(String(error)); } finally { setBusy(false); } };
  return <><PageHead title="销量预测" actions={<Tag color="green">推荐模型：跨门店增强 Two-stage（E2）</Tag>} /><Card>{ready.length ? <Form form={form} layout="vertical" initialValues={{ model_ids: ['E2'] }}><Form.Item label="选择逻辑数据集"><Select value={selected?.dataset_id} onChange={setSelectedDatasetId} options={ready.map(d => ({ value: d.dataset_id, label: `${d.dataset_name}（${d.date_range.start} 至 ${d.date_range.end}）` }))} /></Form.Item>{selected && <Card size="small" className="source-card"><Descriptions size="small" column={2}><Descriptions.Item label="当前数据">{selected.dataset_name}</Descriptions.Item><Descriptions.Item label="覆盖区间">{selected.date_range.start} 至 {selected.date_range.end}</Descriptions.Item><Descriptions.Item label="数据截止月份">{observation}</Descriptions.Item><Descriptions.Item label="预测目标月份">{target}</Descriptions.Item></Descriptions></Card>}<Form.Item name="model_ids" label="选择模型" rules={[{ required: true, message: '请选择模型' }]}><Checkbox.Group options={[...runnableModels.map(id => ({ value: id, label: modelNames[id] })), { value: 'RF', label: '随机森林（暂未接入当前推理）', disabled: true }, { value: 'LGBM', label: 'LightGBM（历史实验模型）', disabled: true }, { value: 'MLP', label: 'MLP（历史实验模型）', disabled: true }]} /></Form.Item>{busy && <Alert type="info" showIcon message="预测运行中" description="后端正在执行真实预测任务；当前接口返回单一进度值，不展示伪造分阶段完成度。" />}<Button className="action-row" type="primary" loading={busy} onClick={() => void submit()}>开始预测</Button></Form> : <Empty description="暂无就绪数据集，请先完成数据接入。" />}</Card></>;
}

function PredictionPage({ refresh, onNavigate }: { datasets: Dataset[]; refresh: () => Promise<void>; onNavigate: (view: View) => void }) {
  const [form] = Form.useForm();
  const [busy, setBusy] = useState(false);
  const [readiness, setReadiness] = useState<PredictionReadiness>();
  const { data: months } = useCachedData<DataMonthCatalog>('data-months', getDataMonths, []);
  const latestMonth = months?.summary.date_range?.end;
  const defaultTarget = nextMonth(latestMonth);
  const values = Form.useWatch([], form);
  useEffect(() => {
    if (!form.getFieldValue('target_month') && latestMonth) form.setFieldValue('target_month', defaultTarget);
  }, [defaultTarget, form, latestMonth]);
  useEffect(() => {
    const target = values?.target_month || defaultTarget;
    const modelIds = values?.model_ids || ['E2'];
    if (!target || !modelIds.length || target === '暂无') return;
    checkPredictionReadiness({ target_month: target, model_ids: modelIds }).then(setReadiness).catch(error => setReadiness({ status: 'NOT_READY', target_month: target, observation_month: nextMonth(undefined), detail: String(error) }));
  }, [defaultTarget, values?.target_month, JSON.stringify(values?.model_ids ?? ['E2'])]);
  const submit = async () => {
    const formValues = await form.validateFields();
    setBusy(true);
    try {
      const checked = await checkPredictionReadiness({ target_month: formValues.target_month, model_ids: formValues.model_ids });
      setReadiness(checked);
      if (checked.status !== 'READY') {
        message.error('数据准备未完成，不能发起预测。');
        return;
      }
      await createPrediction({ target_month: formValues.target_month, model_ids: formValues.model_ids });
      await refresh();
      message.success('预测完成，已生成未来月份结果。');
      onNavigate('analysis');
    } catch (error) {
      message.error(String(error));
    } finally {
      setBusy(false);
    }
  };
  return <><PageHead title="销量预测" actions={<Tag color="green">目标月份驱动</Tag>} /><Card><Form form={form} layout="vertical" initialValues={{ target_month: defaultTarget, model_ids: ['E2'] }}><Form.Item name="target_month" label="预测目标月份" rules={[{ required: true, message: '请输入预测目标月份' }]}><Input placeholder="例如：2026-07" /></Form.Item><Form.Item name="model_ids" label="选择模型" rules={[{ required: true, message: '请选择模型' }]}><Checkbox.Group options={[...runnableModels.map(id => ({ value: id, label: modelNames[id] })), { value: 'RF', label: '随机森林（暂未接入当前推理）', disabled: true }, { value: 'LGBM', label: 'LightGBM（历史实验模型）', disabled: true }, { value: 'MLP', label: 'MLP（历史实验模型）', disabled: true }]} /></Form.Item>{readiness && <Card size="small" className="source-card"><Descriptions size="small" column={2}><Descriptions.Item label="数据截止月份">{readiness.observation_month}</Descriptions.Item><Descriptions.Item label="预测目标月份">{readiness.target_month}</Descriptions.Item><Descriptions.Item label="检查状态">{readiness.status === 'READY' ? statusTag('SUCCESS') : statusTag('FAILED')}</Descriptions.Item><Descriptions.Item label="所需月份">{readiness.required_months?.join('、') ?? '待检查'}</Descriptions.Item></Descriptions>{readiness.status !== 'READY' && <Alert type="error" showIcon message="数据准备失败" description={`缺少历史数据：${readiness.missing_months?.join('、') || readiness.detail || '未知原因'}；影响特征：${readiness.affected_features?.slice(0, 5).join('、') || '模型输入特征'}`} />}</Card>}<Button className="action-row" type="primary" loading={busy} disabled={readiness?.status === 'NOT_READY'} onClick={() => void submit()}>开始预测</Button></Form></Card></>;
}

function AnalysisPage({ runs }: { runs: PredictionRun[] }) {
  const successRuns = runs.filter(item => item.status === 'SUCCESS'); const [selectedRunId, setSelectedRunId] = useState<string>(); const run = successRuns.find(item => item.prediction_run_id === selectedRunId) ?? successRuns[0]; const [model, setModel] = useState('E2');
  const { data: summary } = useCachedData(run ? `summary:${run.prediction_run_id}` : undefined, () => getPredictionSummary(run!.prediction_run_id), [run?.prediction_run_id]);
  useEffect(() => { if (run && !run.model_ids.includes(model)) setModel(run.model_ids[0]); }, [run, model]);
  const current = model ? summary?.model_summaries[model] : undefined;
  return <><PageHead title="预测分析" actions={run && <Space><Button href={predictionExcelUrl(run.prediction_run_id, { modelId: model, topN: 100 })}>导出结果</Button><NotificationButton run={run} modelId={model} /></Space>} />{!run && <Empty className="page-empty" description="暂无预测结果，请先完成数据接入并发起预测。" />}{run && <Card className="run-head"><div><div className="analysis-title">预测目标月份：{targetMonth(summary?.observation_month ?? run.observation_month, summary?.target_month)}</div><div className="analysis-meta">数据截止月份：{summary?.observation_month ?? run.observation_month ?? '暂无'} | 模型：{modelNames[model] ?? model} | 任务创建时间：{displayTime(run.created_at)} | 状态：成功</div><Text type="secondary">技术任务 ID：{run.prediction_run_id}</Text></div><Select className="run-select" value={run.prediction_run_id} onChange={setSelectedRunId} options={successRuns.map(item => ({ value: item.prediction_run_id, label: runLabel(item, item.prediction_run_id === run.prediction_run_id ? summary : undefined) }))} /></Card>}{current && <Row gutter={[12, 12]}><Col span={5}><MetricCard title="预测总销量" value={formatNumber(current.prediction_total)} /></Col><Col span={5}><MetricCard title="预计动销图书" value={formatNumber(current.predicted_nonzero_book_count)} /></Col><Col span={5}><MetricCard title="高动销图书" value={formatNumber(current.mc_counts.MC4 ?? 0)} /></Col><Col span={5}><MetricCard title="无动销图书" value={formatNumber(current.mc_counts.MC0 ?? 0)} /></Col><Col span={4}><MetricCard title="覆盖门店" value={formatNumber(current.store_count)} /></Col></Row>}{run && <Tabs className="analysis-tabs" destroyInactiveTabPane={false} items={[{ key: 'overview', label: '总体概览', children: <><Space wrap className="filter-row"><Text>当前模型</Text><Select value={model} onChange={setModel} options={run.model_ids.map(id => ({ value: id, label: modelNames[id] ?? id }))} /></Space><Row gutter={[16, 16]}><Col span={12}><PredictionTrend run={run} modelId={model} /></Col><Col span={12}><McChart summary={current} /></Col><Col span={12}><TopBooksPanel run={run} modelId={model} /></Col><Col span={12}><DifficultOverview run={run} modelId={model} /></Col></Row><ResultTable run={run} defaultModel={model} /></> }, { key: 'store', label: '门店分析', children: <StoreAnalysis run={run} modelId={model} onModelChange={setModel} /> }, { key: 'book', label: '图书分析', children: <BookAnalysis run={run} modelId={model} onModelChange={setModel} /> }, { key: 'metrics', label: '模型对比', children: <Card title="模型验证指标"><Empty description="暂无正式验证指标。当前页面不使用预测总量冒充模型效果。" /></Card> }]} />}</>;
}

const baseResultColumns: ColumnsType<Record<string, unknown>> = [{ title: '门店', dataIndex: 'site_no' }, { title: '图书编号', dataIndex: 'item_id', sorter: true }, { title: '图书名称', dataIndex: 'book_name' }, { title: '预测销量', dataIndex: 'pred_qty_int', sorter: true }, { title: '动销等级', dataIndex: 'pred_mc', render: (v: string) => mcLabels[v] || v }, { title: '动销概率', dataIndex: 'p_sale', render: fixed, sorter: true }, { title: '动销后预计销量', dataIndex: 'conditional_qty', render: fixed }];

function ResultTable({ run, defaultModel }: { run: PredictionRun; defaultModel: string }) {
  const [model, setModel] = useState(defaultModel); const [site, setSite] = useState(''); const [mc, setMc] = useState<string>(); const [keyword, setKeyword] = useState(''); const [sortBy, setSortBy] = useState<SortBy>('pred_qty_int'); const [sortOrder, setSortOrder] = useState<SortOrder>('desc'); const [rows, setRows] = useState<PagedResults>();
  const load = (page = 1) => cached<PagedResults>(`detail:${run.prediction_run_id}:${model}:${site}:${mc ?? ''}:${page}:${sortBy}:${sortOrder}`, () => getPredictionResults(run.prediction_run_id, 'predictions', { modelId: model, siteNo: site || undefined, mc, page, pageSize: 50, sortBy, sortOrder })).then(setRows);
  useEffect(() => { setModel(defaultModel); }, [defaultModel]);
  useEffect(() => { void load(1); }, [run.prediction_run_id, model, site, mc, sortBy, sortOrder]);
  const filteredItems = (rows?.items ?? []).filter(row => !keyword || String(row.item_id).includes(keyword) || String(row.book_name ?? '').includes(keyword));
  const onChange = (pagination: TablePaginationConfig, _filters: unknown, sorter: any) => { if (sorter?.field && ['pred_qty_int', 'p_sale', 'item_id'].includes(sorter.field)) { setSortBy(sorter.field); setSortOrder(sorter.order === 'ascend' ? 'asc' : 'desc'); } else { void load(pagination.current ?? 1); } };
  return <Card title="预测明细表" className="table-card"><Space wrap className="filter-row"><Text strong>明细筛选</Text><Select value={model} onChange={setModel} options={run.model_ids.map(id => ({ value: id, label: modelNames[id] ?? id }))} /><Input value={site} onChange={e => setSite(e.target.value)} placeholder="门店编号" /><Select allowClear value={mc} onChange={setMc} placeholder="动销等级" options={Object.entries(mcLabels).map(([value, label]) => ({ value, label }))} /><Input value={keyword} onChange={e => setKeyword(e.target.value)} placeholder="图书名称 / 编号" /></Space><Text type="secondary">默认排序：预测销量从高到低 → 动销概率从高到低 → 图书编号升序。排序在后端分页前执行。</Text><Table rowKey={row => `${row.site_no}-${row.item_id}-${row.model_id}`} dataSource={filteredItems} columns={baseResultColumns} onChange={onChange} pagination={{ current: rows?.page ?? 1, pageSize: rows?.page_size ?? 50, total: rows?.total ?? 0, onChange: page => void load(page) }} /></Card>;
}

function DifficultOverview({ run, modelId }: { run: PredictionRun; modelId: string }) {
  const { data: summary } = useCachedData<DifficultBooksSummary>(`difficult-summary:${run.prediction_run_id}:${modelId}`, () => getDifficultBooksSummary(run.prediction_run_id, { modelId }), [run.prediction_run_id, modelId]);
  return <Card title="历史难预测对象概览">{summary?.difficult_book_count ? <><Statistic title="困难预测记录" value={summary.difficult_book_count} /><Text type="secondary">来源于带 actual_qty 的历史验证预测产物。</Text></> : <Empty description="暂无可用历史验证结果" />}</Card>;
}

function StoreAnalysis({ run, modelId, onModelChange }: { run: PredictionRun; modelId: string; onModelChange: (v: string) => void }) {
  const [siteNo, setSiteNo] = useState(''); const { data: stores } = useCachedData<PagedResults>(`stores:${run.prediction_run_id}:${modelId}:100`, () => getPredictionResults(run.prediction_run_id, 'store_summary', { modelId, pageSize: 100 }), [run.prediction_run_id, modelId]); const selected = siteNo || String(stores?.items[0]?.site_no ?? ''); const { data: summary } = useCachedData<Record<string, number>>(selected ? `store-summary:${run.prediction_run_id}:${modelId}:${selected}` : undefined, () => getStorePredictionSummary(run.prediction_run_id, selected, modelId), [run.prediction_run_id, modelId, selected]);
  return <><Card title="门店筛选"><Space><Select value={modelId} onChange={onModelChange} options={run.model_ids.map(id => ({ value: id, label: modelNames[id] ?? id }))} /><Select showSearch value={selected || undefined} placeholder="选择真实门店" onChange={setSiteNo} options={(stores?.items ?? []).map(row => ({ value: String(row.site_no), label: `门店编号 ${row.site_no}` }))} /></Space></Card>{selected && <Row gutter={[16, 16]} className="section-grid"><Col span={8}><MetricCard title="预计总销量" value={formatNumber(summary?.prediction_total)} /></Col><Col span={8}><MetricCard title="预计动销图书数" value={formatNumber(summary?.pred_nonzero_count)} /></Col><Col span={8}><MetricCard title="高动销图书数" value={formatNumber((summary?.pred_mc3_count ?? 0) + (summary?.pred_mc4_count ?? 0))} /></Col><Col span={12}><PredictionTrend run={run} modelId={modelId} siteNo={selected} /></Col><Col span={12}><TopBooksPanel run={run} modelId={modelId} siteNo={selected} /></Col></Row>}</>;
}

function BookAnalysis({ run, modelId, onModelChange }: { run: PredictionRun; modelId: string; onModelChange: (v: string) => void }) {
  const [site, setSite] = useState(''); const [keyword, setKeyword] = useState(''); const [rows, setRows] = useState<Record<string, unknown>[]>([]);
  useEffect(() => { if (!keyword) { setRows([]); return; } cached<PagedResults>(`book-search:${run.prediction_run_id}:${modelId}:${site}:${keyword}`, () => getPredictionResults(run.prediction_run_id, 'predictions', { modelId, siteNo: site || undefined, pageSize: 200 })).then(result => setRows(result.items.filter(row => String(row.item_id).includes(keyword) || String(row.book_name ?? '').includes(keyword)).slice(0, 20))).catch(() => setRows([])); }, [run.prediction_run_id, modelId, site, keyword]);
  return <Card title="图书查询"><Space className="filter-row"><Select value={modelId} onChange={onModelChange} options={run.model_ids.map(id => ({ value: id, label: modelNames[id] ?? id }))} /><Input value={site} onChange={e => setSite(e.target.value)} placeholder="门店编号" /><Input value={keyword} onChange={e => setKeyword(e.target.value)} placeholder="搜索图书名称 / 图书编号" /></Space>{rows.length ? <Table rowKey={row => `${row.site_no}-${row.item_id}`} dataSource={rows} columns={baseResultColumns.filter(col => col.title !== '动销后预计销量')} pagination={false} expandable={{ expandedRowRender: row => <><Descriptions size="small" column={3}><Descriptions.Item label="动销概率 p_sale">{fixed(row.p_sale)}</Descriptions.Item><Descriptions.Item label="动销后预计销量 conditional_qty">{fixed(row.conditional_qty)}</Descriptions.Item><Descriptions.Item label="Two-stage">先预测是否产生销量，再估计动销条件下的销量。</Descriptions.Item></Descriptions><PredictionTrend run={run} modelId={modelId} siteNo={String(row.site_no)} itemId={String(row.item_id)} /></> }} /> : <Empty description={keyword ? '未找到匹配图书' : '请输入图书名称或编号查询'} />}</Card>;
}

function DifficultBooksPageView({ runs }: { runs: PredictionRun[] }) {
  const run = runs.find(item => item.status === 'SUCCESS'); const [modelId, setModelId] = useState('E2'); const [siteNo, setSiteNo] = useState(''); const [level, setLevel] = useState<string>(); const key = run ? `difficult:${run.prediction_run_id}:${modelId}:${siteNo}:${level ?? ''}` : undefined; const { data: rows } = useCachedData<DifficultBooksPage>(key, () => getDifficultBooks(run!.prediction_run_id, { modelId, siteNo: siteNo || undefined, difficultyLevel: level, page: 1, pageSize: 50 }), [key]); const { data: summary } = useCachedData<DifficultBooksSummary>(run ? `difficult-summary:${run.prediction_run_id}:${modelId}:${siteNo}:${level ?? ''}` : undefined, () => getDifficultBooksSummary(run!.prediction_run_id, { modelId, siteNo: siteNo || undefined, difficultyLevel: level }), [run?.prediction_run_id, modelId, siteNo, level]);
  const hasData = Boolean(summary?.difficult_book_count);
  return <><PageHead title="难预测图书" actions={run && <Space><ApiSchemaButton schema={summary?.field_schema ?? rows?.field_schema} />{hasData && <Button type="primary" href={difficultBooksExportUrl(run.prediction_run_id, { modelId, siteNo: siteNo || undefined, difficultyLevel: level })}>导出研究数据</Button>}</Space>} />{!run && <Empty description="暂无预测结果，无法识别历史验证难预测对象。" />}{run && !hasData && <Empty className="page-empty" description="暂无可用历史验证结果。完成模型历史验证后，系统将自动生成难预测对象清单。" />}{run && hasData && <><Card className="source-card"><Descriptions size="small" column={4}><Descriptions.Item label="来源模型">{modelNames[modelId]}</Descriptions.Item><Descriptions.Item label="验证来源">历史 Validation / Rolling Backtest 预测产物</Descriptions.Item><Descriptions.Item label="验证样本量">{summary?.difficult_book_count}</Descriptions.Item><Descriptions.Item label="阈值状态">待项目组确认</Descriptions.Item></Descriptions></Card><Row gutter={[12, 12]}><Col span={5}><MetricCard title="困难预测记录" value={formatNumber(summary?.difficult_book_count)} /></Col><Col span={5}><MetricCard title="难预测占比" value={percent(summary?.difficult_ratio)} /></Col><Col span={5}><MetricCard title="平均绝对误差" value={summary?.avg_abs_error?.toFixed(2) ?? '暂无'} /></Col><Col span={5}><MetricCard title="涉及门店" value={formatNumber(summary?.store_count)} /></Col><Col span={4}><MetricCard title="困难级对象" value={formatNumber(summary?.hard_count)} /></Col></Row><Card title="困难预测记录清单" className="table-card"><Alert type="info" showIcon message={summary?.difficulty_rules.note} /><Space wrap className="filter-row"><Select value={modelId} onChange={setModelId} options={runnableModels.map(id => ({ value: id, label: modelNames[id] }))} /><Input value={siteNo} onChange={e => setSiteNo(e.target.value)} placeholder="门店编号" /><Select allowClear value={level} onChange={setLevel} placeholder="困难等级" options={['困难', '较难'].map(value => ({ value, label: value }))} /></Space><Table rowKey={row => `${row.site_no}-${row.item_id}-${row.model_id}`} dataSource={rows?.items ?? []} columns={[{ title: '门店编号', dataIndex: 'site_no' }, { title: '图书编号', dataIndex: 'item_id' }, { title: '图书名称', dataIndex: 'book_name' }, { title: '真实销量', dataIndex: 'actual_qty' }, { title: '预测销量', dataIndex: 'pred_qty' }, { title: '绝对误差', dataIndex: 'abs_error' }, { title: '稳定比例误差', dataIndex: 'stable_error', render: fixed }, { title: '来源模型', dataIndex: 'model_id', render: (v: string) => modelNames[v] ?? v }, { title: '动销等级', dataIndex: 'pred_mc', render: (v: string) => mcLabels[v] || v }, { title: '困难等级', dataIndex: 'difficulty_level', render: (v: string) => <Tag color={v === '困难' ? 'red' : 'orange'}>{v}</Tag> }]} /></Card></>}</>;
}

function ApiSchemaButton({ schema }: { schema?: { field: string; label: string; description: string }[] }) {
  const [open, setOpen] = useState(false);
  return <><Button onClick={() => setOpen(true)}>查看接口字段</Button><Modal title="难预测图书预留接口字段" open={open} onCancel={() => setOpen(false)} footer={null} width={760}><Table rowKey="field" size="small" pagination={false} dataSource={schema ?? []} columns={[{ title: '字段', dataIndex: 'field' }, { title: '业务含义', dataIndex: 'label' }, { title: '说明', dataIndex: 'description' }]} /></Modal></>;
}

function ModelsPage() {
  const [selected, setSelected] = useState<typeof modelDetails[number]>();
  const groups = useMemo(() => modelDetails.reduce<Record<string, typeof modelDetails>>((acc, item) => { acc[item.category] = [...(acc[item.category] ?? []), item]; return acc; }, {}), []);
  return <><PageHead title="模型说明" actions={<Tag color="green">当前推荐模型：跨门店增强 Two-stage（E2）</Tag>} /><Row gutter={[16, 16]}>{Object.entries(groups).map(([category, items]) => <Col span={12} key={category}><Card title={category}>{items.map(item => <button className="model-row model-button" key={item.name} onClick={() => setSelected(item)}><div><Text strong>{item.name}</Text><div className="muted">{item.tech}</div></div><Tag color={item.formal ? 'green' : item.status === '实验对照' ? 'blue' : 'default'}>{item.status}</Tag></button>)}</Card></Col>)}</Row><Drawer title={selected?.name} open={Boolean(selected)} onClose={() => setSelected(undefined)} width={520}>{selected && <Descriptions column={1} bordered size="small"><Descriptions.Item label="模型是什么">{selected.tech}</Descriptions.Item><Descriptions.Item label="为什么用于本项目">{selected.why}</Descriptions.Item><Descriptions.Item label="基本预测流程">{selected.flow}</Descriptions.Item><Descriptions.Item label="当前状态">{selected.status}</Descriptions.Item><Descriptions.Item label="可用于正式预测">{selected.formal ? '是' : '否'}</Descriptions.Item><Descriptions.Item label="关键输出">{selected.outputs}</Descriptions.Item><Descriptions.Item label="真实验证指标">暂无正式验证指标接口，待接入。</Descriptions.Item></Descriptions>}</Drawer></>;
}

function TasksPage({ jobs, refresh }: { jobs: Job[]; refresh: () => Promise<void> }) {
  const [selected, setSelected] = useState<Job>(); const typeLabel: Record<string, string> = { PREDICTION: '预测任务', DATA_PROCESSING: '数据处理', UPLOAD: '文件上传' }; const content = (job: Job) => job.input_files.map(path => path.split(/[\\/]/).pop()).join('；') || '暂无';
  return <><PageHead title="任务管理" actions={<Button onClick={() => void refresh()}>刷新</Button>} /><Card><Table rowKey="job_id" dataSource={jobs} columns={[{ title: '业务任务编号', render: (_, row: Job) => `${(typeLabel[row.job_type] ?? row.job_type).slice(0, 2).toUpperCase()}-${row.job_id.slice(0, 8)}` }, { title: '任务类型', dataIndex: 'job_type', render: (v: string) => typeLabel[v] ?? v }, { title: '任务内容', render: (_, row: Job) => content(row) }, { title: '开始时间', dataIndex: 'started_at', render: displayTime }, { title: '完成时间', dataIndex: 'finished_at', render: displayTime }, { title: '当前状态', dataIndex: 'status', render: statusTag }, { title: '操作', render: (_, row: Job) => <Button type="link" onClick={() => setSelected(row)}>查看详情</Button> }]} /></Card><Drawer title="任务详情" open={Boolean(selected)} onClose={() => setSelected(undefined)} width={620}>{selected && <Descriptions column={1} bordered size="small"><Descriptions.Item label="业务任务编号">{`${(typeLabel[selected.job_type] ?? selected.job_type).slice(0, 2).toUpperCase()}-${selected.job_id.slice(0, 8)}`}</Descriptions.Item><Descriptions.Item label="完整技术任务 ID">{selected.job_id}</Descriptions.Item><Descriptions.Item label="任务类型">{typeLabel[selected.job_type] ?? selected.job_type}</Descriptions.Item><Descriptions.Item label="任务内容">{content(selected)}</Descriptions.Item><Descriptions.Item label="输入文件">{selected.input_files.join('；') || '无'}</Descriptions.Item><Descriptions.Item label="输出文件">{selected.output_files.join('；') || '无'}</Descriptions.Item><Descriptions.Item label="开始时间">{displayTime(selected.started_at)}</Descriptions.Item><Descriptions.Item label="完成时间">{displayTime(selected.finished_at)}</Descriptions.Item><Descriptions.Item label="当前状态">{statusTag(selected.status)}</Descriptions.Item><Descriptions.Item label="错误信息">{selected.error_message ?? '无'}</Descriptions.Item><Descriptions.Item label="日志信息">{selected.log_path}</Descriptions.Item></Descriptions>}</Drawer></>;
}

function NotificationButton({ run, modelId, siteNo, mc }: { run: PredictionRun; modelId: string; siteNo?: string; mc?: string }) {
  const [open, setOpen] = useState(false); const [email, setEmail] = useState(''); const [includeExcel, setIncludeExcel] = useState(true); const [busy, setBusy] = useState(false);
  const send = async () => { setBusy(true); try { const record = await sendPredictionNotification(run.prediction_run_id, { target_email: email, model_id: modelId, site_no: siteNo, mc, include_excel_link: includeExcel }); record.status === 'SUCCESS' ? message.success('邮件已发送') : message.error(record.error_message || '邮件发送失败'); setOpen(false); } catch (error) { message.error(String(error)); } finally { setBusy(false); } };
  return <><Button onClick={() => setOpen(true)}>发送结果</Button><Modal title="发送预测结果" open={open} onCancel={() => setOpen(false)} onOk={() => void send()} okButtonProps={{ loading: busy, disabled: !email }}><Input value={email} onChange={e => setEmail(e.target.value)} placeholder="收件邮箱" /><Checkbox checked={includeExcel} onChange={e => setIncludeExcel(e.target.checked)} className="action-row">包含 Excel 导出链接</Checkbox></Modal></>;
}

function NotificationsPage() {
  const { data: records } = useCachedData<{ items: NotificationRecord[] }>('notifications', getNotifications, []); const { data: smtp } = useCachedData<SmtpStatus>('smtp-status', getSmtpStatus, []); const { data: initial } = useCachedData<NotificationSettings>('notification-settings', getNotificationSettings, []); const [form] = Form.useForm(); const [busy, setBusy] = useState(false); const [testBusy, setTestBusy] = useState(false); const [testEmail, setTestEmail] = useState('');
  useEffect(() => { if (initial) form.setFieldsValue({ target_email: initial.target_email, enabled_types: initial.enabled_types }); }, [initial, form]);
  const save = async () => { const values = await form.validateFields(); setBusy(true); try { await saveNotificationSettings(values); clearCache(); message.success('通知设置已保存'); } finally { setBusy(false); } };
  const sendTest = async () => { setTestBusy(true); try { const result = await sendTestEmail(testEmail); clearCache(); result.status === 'SUCCESS' ? message.success('测试邮件已发送') : message.error(result.error_message || '邮件发送失败'); } finally { setTestBusy(false); } };
  return <><PageHead title="通知设置" /><Card title="正式通知设置"><Descriptions size="small" column={1} className="source-card"><Descriptions.Item label="邮件服务">{smtp?.status ?? '加载中'}</Descriptions.Item></Descriptions><Form form={form} layout="vertical" initialValues={{ enabled_types: ['PREDICTION_SUCCESS', 'PREDICTION_FAILED', 'DATA_PROCESSING_FAILED'] }}><Form.Item name="target_email" label="收件邮箱" rules={[{ required: true, message: '请输入收件邮箱' }]}><Input placeholder="name@example.com" /></Form.Item><Form.Item name="enabled_types" label="通知类型"><Checkbox.Group options={notificationTypeOptions} /></Form.Item><Button type="primary" loading={busy} onClick={() => void save()}>保存通知设置</Button></Form><Collapse className="section-grid" items={[{ key: 'advanced', label: '高级设置', children: <Space direction="vertical" className="full-width"><Alert type="info" showIcon message="SMTP 配置" description="邮件服务读取后端环境变量或 software/config/app.yaml 中的 SMTP 配置，密码建议仅通过环境变量提供。" /><Space><Input value={testEmail} onChange={e => setTestEmail(e.target.value)} placeholder="测试收件邮箱" /><Button loading={testBusy} disabled={!testEmail} onClick={() => void sendTest()}>发送测试邮件</Button></Space></Space> }]} /></Card><Card className="table-card" title="最近通知"><Table rowKey="notification_id" dataSource={records?.items ?? []} locale={{ emptyText: '暂无通知记录，这不代表邮件服务不可用。' }} columns={[{ title: '类型', dataIndex: 'type' }, { title: '收件人', dataIndex: 'target_email' }, { title: '状态', dataIndex: 'status', render: statusTag }, { title: '关联预测', dataIndex: 'related_run_id' }, { title: '创建时间', dataIndex: 'created_at', render: displayTime }, { title: '错误', dataIndex: 'error_message', render: (v: string | null) => v ?? '无' }]} /></Card></>;
}
