from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_pipeline_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if "paths" not in config:
        raise KeyError(
            f"Config {config_path} is not an MRI AD/CN pipeline config: missing top-level 'paths' section. "
            "For the MRI AD/CN 96 pipeline use configs/oasis3_mri_ad_cn_ablation_96.yaml."
        )
    config["_config_path"] = str(config_path.resolve())
    config["_local_repo_root"] = str(REPO_ROOT)
    return config


def local_path(config: dict[str, Any], value: str | Path) -> Path:
    candidate = Path(os.path.expandvars(str(value))).expanduser()
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def cfg_path(config: dict[str, Any], key: str) -> Path:
    if key not in config.get("paths", {}):
        loaded = config.get("_config_path", "<unknown config>")
        available = sorted(config.get("paths", {}).keys())
        raise KeyError(
            f"Config {loaded} is missing paths.{key}. "
            f"Available path keys: {available}. "
            "For this job use configs/oasis3_mri_ad_cn_ablation_96.yaml."
        )
    return local_path(config, config["paths"][key])


def output_dir(config: dict[str, Any], *parts: str) -> Path:
    return cfg_path(config, "output_root").joinpath(*parts)


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def html_escape(value: object) -> str:
    import html

    return html.escape("" if value is None else str(value))
