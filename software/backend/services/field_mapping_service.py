from __future__ import annotations

from .schema_detection_service import REQUIRED_FIELDS, mapping_needs_confirmation


def validate_mapping(mapping: dict[str, str | None], headers: list[str]) -> None:
    missing = [field for field in REQUIRED_FIELDS if not mapping.get(field)]
    unknown = [value for value in mapping.values() if value is not None and value not in headers]
    if missing:
        raise ValueError(f"Field mapping requires confirmation for: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"Mapped source columns not found: {unknown}")


def mapping_status(mapping: dict[str, str | None]) -> str:
    return "NEEDS_CONFIRMATION" if mapping_needs_confirmation(mapping) else "READY"
