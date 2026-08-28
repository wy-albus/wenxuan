# 文轩集团图书销量预测系统前端

## 本地开发

先在项目根目录启动独立后端：

```powershell
python -m uvicorn software.backend.api.main:app --reload
```

然后启动前端：

```powershell
cd software/frontend
npm install
npm run dev
```

复制 `.env.example` 为 `.env.local`，并设置 `VITE_API_BASE_URL` 为后端地址。

## GitHub Pages

GitHub Pages 只部署静态前端；FastAPI、数据处理、模型和预测仍须在本地或云端单独运行。工作流读取 GitHub Actions Variables：

- `VITE_API_BASE_URL`：可从浏览器访问的后端 HTTPS 地址。
- `PAGES_BASE_PATH`：项目页设为 `/仓库名/`；`username.github.io` 根站点设为 `/`。

后端须将 Pages 地址添加到 `CORS_ORIGINS`，例如：

```powershell
$env:CORS_ORIGINS='https://username.github.io'
python -m uvicorn software.backend.api.main:app --reload
```
