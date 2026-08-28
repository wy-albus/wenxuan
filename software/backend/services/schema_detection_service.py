from __future__ import annotations

import csv
from pathlib import Path


FIELD_ALIASES = {
    "site_no": ("site_no", "site", "store_id", "store_no", "门店编码", "门店编号", "门店号", "店号"),
    "item_id": ("item_id", "book_id", "gds_no", "product_id", "商品编码", "图书编码", "书号"),
    "sale_date": ("sale_date", "date", "period", "sales_date", "交易日期", "销售日期", "日期"),
    "qty": ("qty", "quantity", "sales_qty", "count", "数量", "销量", "销售数量"),
    "isbn": ("isbn", "ISBN", "国际标准书号"),
    "book_name": ("book_name", "title", "name", "书名", "图书名称", "商品名称"),
    "category": ("category", "category_name", "分类", "品类", "图书分类"),
    "amount": ("amount", "sales_amount", "tsp", "实洋", "销售额", "金额"),
    "price": ("price", "unit_price", "定价", "售价", "价格"),
    "channel": ("channel", "oln_or_ofln", "渠道", "销售渠道", "线上线下"),
}
REQUIRED_FIELDS = ("site_no", "item_id", "sale_date", "qty")


def read_csv_headers(path: Path) -> list[str]:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                return next(csv.reader(handle))
        except (UnicodeDecodeError, StopIteration):
            continue
    raise ValueError(f"Cannot read CSV header: {path.name}")


def detect_field_mapping(headers: list[str]) -> dict[str, str | None]:
    normalized = {str(header).strip().lower(): header for header in headers}
    result: dict[str, str | None] = {}
    for field, aliases in FIELD_ALIASES.items():
        result[field] = next((normalized[alias.lower()] for alias in aliases if alias.lower() in normalized), None)
    if result["item_id"] is None and result["isbn"] is not None:
        result["item_id"] = result["isbn"]
    return result


def mapping_needs_confirmation(mapping: dict[str, str | None]) -> bool:
    return any(mapping[field] is None for field in REQUIRED_FIELDS)
