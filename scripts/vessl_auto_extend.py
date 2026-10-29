#!/usr/bin/env python3
"""Extend a VESSL workspace shortly before its scheduled termination."""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("vessl_auto_extend")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor and extend a VESSL workspace before it expires."
    )
    parser.add_argument("--organization", required=True)
    parser.add_argument("--workspace-id", required=True, type=int)
    parser.add_argument(
        "--max-total-hours",
        type=int,
        help="Optional local safety cap. By default, rely on the VESSL server limit.",
    )
    parser.add_argument(
        "--extend-hours",
        type=int,
        default=24,
        help="Hours requested per extension (default: 24).",
    )
    parser.add_argument(
        "--threshold-minutes",
        type=int,
        default=30,
        help="Extend when remaining time is at or below this value (default: 30).",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=300,
        help="Polling interval with --watch (default: 300).",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Keep monitoring until the workspace stops or reaches the cap.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually extend the workspace. Without this flag, run as a dry run.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Log path (default: results/logs/vessl-auto-extend-<workspace-id>.log).",
    )
    args = parser.parse_args()

    for field in ("extend_hours", "threshold_minutes", "poll_seconds"):
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be greater than zero")
    if args.max_total_hours is not None and args.max_total_hours <= 0:
        parser.error("--max-total-hours must be greater than zero")
    if args.log_file is None:
        args.log_file = Path(
            f"results/logs/vessl-auto-extend-{args.workspace_id}.log"
        )
    return args


def configure_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S%z"
    )
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    for handler in (logging.StreamHandler(), logging.FileHandler(log_file)):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def load_vessl() -> tuple[Any, type[Any]]:
    try:
        from vessl import vessl_api
        from vessl.openapi_client.models import WorkspaceExtendRunningTimeAPIInput
    except ImportError as error:
        raise RuntimeError(
            "VESSL SDK is unavailable. Install it with `python -m pip install vessl`."
        ) from error
    return vessl_api, WorkspaceExtendRunningTimeAPIInput


def format_remaining(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def check_once(args: argparse.Namespace, api: Any, request_type: type[Any]) -> bool:
    workspace = api.workspace_read_api(args.organization, args.workspace_id)
    if workspace.status != "running":
        LOGGER.info("Workspace status is %r; monitoring stopped", workspace.status)
        return False

    if workspace.end_dt is None or workspace.max_running_hours is None:
        raise RuntimeError("Workspace response does not include runtime information.")

    now = datetime.now(timezone.utc)
    remaining_seconds = (workspace.end_dt - now).total_seconds()
    elapsed_seconds = (now - workspace.created_dt).total_seconds()
    current_hours = int(workspace.max_running_hours)
    runtime_limit = (
        f"{current_hours}/{args.max_total_hours}h"
        if args.max_total_hours is not None
        else f"{current_hours}h/server-limit"
    )
    LOGGER.info(
        "status=running elapsed=%s remaining=%s runtime=%s created=%s end=%s",
        format_remaining(elapsed_seconds),
        format_remaining(remaining_seconds),
        runtime_limit,
        workspace.created_dt.isoformat(),
        workspace.end_dt.isoformat(),
    )

    available_hours = None
    if args.max_total_hours is not None:
        available_hours = args.max_total_hours - current_hours
        if available_hours <= 0:
            LOGGER.info("Configured total runtime cap reached; monitoring stopped")
            return False
    if remaining_seconds > args.threshold_minutes * 60:
        return True

    extendable = api.workspace_check_extendable_api(
        args.organization, args.workspace_id
    ).is_extendable
    if not extendable:
        LOGGER.warning("VESSL reports that the workspace is not currently extendable")
        return True

    requested_hours = (
        min(args.extend_hours, available_hours)
        if available_hours is not None
        else args.extend_hours
    )
    if not args.apply:
        LOGGER.info(
            "DRY RUN: would extend by %sh; pass --apply to enable extension",
            requested_hours,
        )
        return False

    request = request_type(running_time_extend_hours=requested_hours)
    updated = api.workspace_extend_running_time_api(
        args.organization,
        args.workspace_id,
        workspace_extend_running_time_api_input=request,
    )
    LOGGER.info(
        "Extended by %sh: runtime=%sh end=%s",
        requested_hours,
        updated.max_running_hours,
        updated.end_dt.isoformat(),
    )
    return (
        args.max_total_hours is None
        or int(updated.max_running_hours) < args.max_total_hours
    )


def main() -> int:
    args = parse_args()
    configure_logging(args.log_file)
    LOGGER.info(
        "Monitor started: organization=%s workspace_id=%s apply=%s watch=%s "
        "extend_hours=%s threshold_minutes=%s max_total_hours=%s",
        args.organization,
        args.workspace_id,
        args.apply,
        args.watch,
        args.extend_hours,
        args.threshold_minutes,
        args.max_total_hours if args.max_total_hours is not None else "server-limit",
    )
    try:
        api, request_type = load_vessl()
        while True:
            try:
                keep_monitoring = check_once(args, api, request_type)
            except Exception:
                LOGGER.exception("VESSL check or extension failed")
                if not args.watch:
                    return 1
                keep_monitoring = True
            if not keep_monitoring:
                break
            if not args.watch:
                break
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        LOGGER.info("Monitoring stopped by user")
    except Exception:
        LOGGER.exception("Monitor failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())