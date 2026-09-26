import argparse
import json
import re
from datetime import datetime
from pathlib import Path


def newest_trainer_state(output_dir: Path) -> Path | None:
    direct = output_dir / "trainer_state.json"
    if direct.exists():
        return direct
    candidates = sorted(output_dir.glob("checkpoint-*/trainer_state.json"))
    return candidates[-1] if candidates else None


def load_trainer_state(output_dir: Path) -> dict:
    state_path = newest_trainer_state(output_dir)
    if state_path is None:
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def parse_log_metrics(log_text: str) -> dict:
    progress_matches = re.findall(r"(\d+)/(\d+)\s*\[", log_text)
    loss_matches = re.findall(r"'loss':\s*([0-9.]+)|loss[:=]\s*([0-9.]+)", log_text)
    runtime_errors = re.findall(r"RuntimeError: (.+)", log_text)
    tracebacks = len(re.findall(r"Traceback \(most recent call last\):", log_text))
    warnings = len(re.findall(r"UserWarning:", log_text))
    progress = None
    if progress_matches:
        preferred = [(int(cur), int(total)) for cur, total in progress_matches if int(total) >= 50]
        cur, total = preferred[-1] if preferred else tuple(map(int, progress_matches[-1]))
        progress = {"current": int(cur), "total": int(total)}
    losses = [a or b for a, b in loss_matches if (a or b)]
    return {
        "progress": progress,
        "losses": losses[-5:],
        "runtime_errors": runtime_errors[-5:],
        "tracebacks": tracebacks,
        "warnings": warnings,
    }


def format_summary(args: argparse.Namespace, trainer_state: dict, log_metrics: dict, checkpoints: list[str]) -> str:
    lines: list[str] = []
    lines.append(f"# Meta-Cog Summary: {args.stage}")
    lines.append("")
    lines.append(f"- time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- stage: {args.stage}")
    lines.append(f"- mode: {args.mode}")
    lines.append(f"- status: {args.status}")
    lines.append(f"- output_dir: {args.output_dir}")
    lines.append(f"- log_file: {args.log_file}")
    if log_metrics["progress"] is not None:
        prog = log_metrics["progress"]
        lines.append(f"- progress: {prog['current']}/{prog['total']}")
    if checkpoints:
        lines.append(f"- checkpoints: {', '.join(checkpoints[-5:])}")
    global_step = trainer_state.get("global_step")
    if global_step is not None:
        lines.append(f"- trainer_global_step: {global_step}")
    if trainer_state.get("epoch") is not None:
        lines.append(f"- trainer_epoch: {trainer_state['epoch']}")
    if trainer_state.get("best_model_checkpoint"):
        lines.append(f"- best_model_checkpoint: {trainer_state['best_model_checkpoint']}")
    lines.append(f"- tracebacks_seen: {log_metrics['tracebacks']}")
    lines.append(f"- warnings_seen: {log_metrics['warnings']}")
    lines.append("")
    lines.append("## Recent Loss")
    lines.append("")
    if log_metrics["losses"]:
        for loss in log_metrics["losses"]:
            lines.append(f"- {loss}")
    else:
        lines.append("- no explicit loss parsed yet")
    lines.append("")
    lines.append("## Recent Errors")
    lines.append("")
    if log_metrics["runtime_errors"]:
        for err in log_metrics["runtime_errors"]:
            lines.append(f"- {err}")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--latest-report", required=True)
    args = parser.parse_args()

    log_path = Path(args.log_file)
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    log_text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
    trainer_state = load_trainer_state(output_dir)
    checkpoints = [p.name for p in sorted(output_dir.glob("checkpoint-*"))] if output_dir.exists() else []
    summary = format_summary(args, trainer_state, parse_log_metrics(log_text), checkpoints)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"{args.stage}_{args.status}_{timestamp}.md"
    report_path.write_text(summary, encoding="utf-8")
    Path(args.latest_report).write_text(summary, encoding="utf-8")
    print(f"[summary] wrote {report_path}")


if __name__ == "__main__":
    main()
