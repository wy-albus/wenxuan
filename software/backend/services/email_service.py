from __future__ import annotations

import json
import os
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path

import yaml

from .notification_service import NotificationService
from .runtime import PROJECT_ROOT


@dataclass(frozen=True)
class EmailConfig:
    host: str | None
    port: int
    user: str | None
    password: str | None
    sender: str | None
    use_tls: bool
    public_base_url: str = "http://127.0.0.1:8000"

    @classmethod
    def from_sources(cls) -> "EmailConfig":
        path = PROJECT_ROOT / "software" / "config" / "app.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        smtp = config.get("smtp", {})
        def value(env_name: str, config_name: str, default=None):
            return os.getenv(env_name, smtp.get(config_name, default))
        tls = value("SMTP_USE_TLS", "use_tls", True)
        return cls(
            host=value("SMTP_HOST", "host"), port=int(value("SMTP_PORT", "port", 587)),
            user=value("SMTP_USER", "user"), password=os.getenv("SMTP_PASSWORD"), sender=value("SMTP_FROM", "from"),
            use_tls=str(tls).lower() in {"1", "true", "yes", "on"},
            public_base_url=os.getenv("APP_PUBLIC_BASE_URL", config.get("public_base_url", "http://127.0.0.1:8000")).rstrip("/"),
        )


class EmailService:
    def __init__(self, config: EmailConfig | None = None, notification_service: NotificationService | None = None, smtp_factory=None) -> None:
        self.config = config or EmailConfig.from_sources()
        self.notifications = notification_service or NotificationService()
        self.smtp_factory = smtp_factory or smtplib.SMTP

    def _send(self, *, notification_type: str, target_email: str, subject: str, body: str, related_run_id: str | None = None) -> dict:
        try:
            if not self.config.host or not self.config.sender:
                raise ValueError("SMTP is not configured: set SMTP_HOST and SMTP_FROM")
            message = EmailMessage()
            message["Subject"] = subject
            message["From"] = self.config.sender
            message["To"] = target_email
            message.set_content(body)
            with self.smtp_factory(self.config.host, self.config.port, timeout=20) as smtp:
                if self.config.use_tls:
                    smtp.starttls()
                if self.config.user:
                    smtp.login(self.config.user, self.config.password or "")
                smtp.send_message(message)
            return self.notifications.record(notification_type=notification_type, target_email=target_email, status="SUCCESS", related_run_id=related_run_id)
        except Exception as exc:
            return self.notifications.record(notification_type=notification_type, target_email=target_email, status="FAILED", related_run_id=related_run_id, error_message=str(exc))

    def send_test_email(self, target_email: str) -> dict:
        return self._send(notification_type="TEST", target_email=target_email, subject="文轩集团图书销量预测系统：测试邮件", body="这是一封测试邮件。SMTP 配置与通知记录功能已被调用。")

    def send_prediction_success(self, run: dict, target_email: str, model_id: str, include_excel_link: bool = True, *, site_no: str | None = None, mc: str | None = None) -> dict:
        artifact = json.loads((Path(run["prediction_dir"]) / "summary.json").read_text(encoding="utf-8"))
        summary = artifact["model_summaries"][model_id]
        filters = [f"model_id={model_id}"]
        if site_no:
            filters.append(f"site_no={site_no}")
        if mc:
            filters.append(f"mc={mc}")
        excel = f"{self.config.public_base_url}/api/predictions/{run['prediction_run_id']}/export-excel?{'&'.join(filters)}"
        parquet = f"{self.config.public_base_url}/api/predictions/{run['prediction_run_id']}/download-parquet"
        body = "\n".join([
            "预测任务已完成。", f"prediction_run_id: {run['prediction_run_id']}", f"dataset_id: {run['dataset_id']}", f"model_id: {model_id}",
            f"observation_month: {artifact['observation_month']}", f"筛选条件: {'；'.join(filters)}",
            f"预测总销量: {summary['prediction_total']}", f"有动销图书数: {summary['predicted_nonzero_book_count']}",
            f"MC3（5–19）图书数: {summary['mc_counts']['MC3']}", f"MC4（20+）图书数: {summary['mc_counts']['MC4']}",
            f"覆盖门店数: {summary['store_count']}", f"覆盖图书数: {summary['item_count']}",
            f"Excel 导出接口: {excel}" if include_excel_link else "Excel 链接：未包含", f"Parquet 下载接口: {parquet}",
        ])
        return self._send(notification_type="PREDICTION_SUCCESS", target_email=target_email, subject="文轩集团图书销量预测系统：预测完成", body=body, related_run_id=run["prediction_run_id"])

    def send_prediction_failure(self, run: dict, target_email: str, log_path: str | None) -> dict:
        body = "\n".join(["预测任务失败。", f"prediction_run_id: {run['prediction_run_id']}", f"dataset_id: {run['dataset_id']}", f"model_id: {', '.join(run['model_ids'])}", f"失败时间: {run['created_at']}", f"error_message: {run.get('error_message') or '未知错误'}", f"log_path: {log_path or '未找到'}"])
        return self._send(notification_type="PREDICTION_FAILED", target_email=target_email, subject="文轩集团图书销量预测系统：预测失败", body=body, related_run_id=run["prediction_run_id"])

    def send_export_success(self, run: dict, target_email: str, *, model_id: str | None, site_no: str | None, excel_path: str) -> dict:
        download = f"{self.config.public_base_url}/api/predictions/{run['prediction_run_id']}/export-excel"
        body = "\n".join(["Excel 导出已完成。", f"run_id: {run['prediction_run_id']}", f"model_id: {model_id or 'all'}", f"site_no: {site_no or 'all'}", f"Excel 文件路径: {excel_path}", f"下载接口: {download}"])
        return self._send(notification_type="EXPORT_SUCCESS", target_email=target_email, subject="文轩集团图书销量预测系统：Excel 导出完成", body=body, related_run_id=run["prediction_run_id"])
