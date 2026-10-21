#!/usr/bin/env python3
"""v4 러너(scripts/run_gemma3_v4.sh)의 단계가 끝날 때마다 결과를 commit 하고 push 한다.

  STEP 1  Teacher FT 와 Student SFT 가 모두 끝났을 때 (학습 이력, 로그)
  STEP 2  KD 가 끝났을 때 (KD 학습 이력, 로그)
  STEP 3  평가·비교·bootstrap 까지 끝났을 때 (4-way 결과, bootstrap, 차트)

10분마다 확인하고, 이미 올린 단계는 다시 커밋하지 않으므로 워크스페이스를 재시작한 뒤 다시 실행해도 된다.
진행 표시줄은 걸러내고 올린다. 체크포인트는 .gitignore 로 제외된다.

사용법:
  setsid nohup .venv-gemma/bin/python scripts/v4_git_watcher.py configs/qwen3_8b_1.7b_3090x4_v4.yaml \
    --label "Qwen3 v4" --title "Qwen3 8B -> 1.7B" >> ~/v4_git_watcher.log 2>&1 &
"""
import argparse
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
POLL_SECONDS = 600
TRAILER = "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
PROGRESS = re.compile(r"\d+/\d+ \[")


def log(message):
    print(f"[{datetime.now():%m-%d %H:%M:%S}] {message}", flush=True)


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)


class Run:
    def __init__(self, config_path, label, title):
        config = yaml.safe_load(Path(config_path).read_text())
        out, run_id = Path(config["output_dir"]), config["run_id"]
        self.logs = REPO / out / "logs" / run_id
        self.ckpt = REPO / out / "checkpoints" / run_id
        self.figs = REPO / out / "figures" / run_id
        self.label, self.title = label, title
        self.teacher_precision = config.get("teacher_quantization") or "bf16"
        self.kd_init = "SFT 초기값" if config.get("distill_student_checkpoint") else "Base 초기값"
        self.branch = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()

    def trained(self, stage):
        return (self.logs / f"{stage}_history.json").is_file() and not (self.ckpt / f"{stage}_last.pt").exists()

    def history_line(self, stage, name):
        history = json.loads((self.logs / f"{stage}_history.json").read_text())
        vals = [e["val_ce_loss"] for e in history]
        minutes = sum(e.get("time", 0) for e in history) / len(history) / 60
        lowest = min(range(len(vals)), key=vals.__getitem__) + 1
        return (f"{name}: val CE {' -> '.join(f'{v:.4f}' for v in vals)}"
                f" (val CE 최저 epoch {lowest}, epoch 당 약 {minutes:.0f}분)")


def already_committed(tag):
    return tag in git("log", "--format=%s", "-100").stdout


def strip_progress(path):
    """tqdm 갱신 줄을 지우고 각 진행 표시줄의 마지막 100% 줄만 남긴다."""
    if not path.is_file():
        return
    lines = path.read_bytes().decode("utf-8", "replace").replace("\r", "\n").split("\n")
    kept = []
    for line in lines:
        if not line.strip() or (PROGRESS.search(line) and "100%|" not in line):
            continue
        if kept and kept[-1] == line:
            continue
        kept.append(line)
    path.write_text("\n".join(kept) + "\n")


def commit_and_push(run, paths, subject, body):
    existing = [str(p.relative_to(REPO)) for p in paths if p.exists()]
    if existing:
        git("add", "--", *existing)
    if not git("diff", "--cached", "--name-only").stdout.strip():
        log(f"nothing to commit for: {subject}")
    else:
        result = git("commit", "-q", "-m", subject + "\n\n" + body + TRAILER)
        if result.returncode != 0:
            log(f"commit failed: {result.stderr.strip()}")
            return False
        log(f"committed: {subject}")
    if git("ls-remote", "--exit-code", "--heads", "origin", run.branch).returncode == 0:
        pull = git("pull", "-q", "--rebase", "--autostash", "origin", run.branch)
        if pull.returncode != 0:
            log(f"pull --rebase failed, will retry: {pull.stderr.strip()}")
            git("rebase", "--abort")
            return False
    push = git("push", "-q", "-u", "origin", f"HEAD:{run.branch}")
    if push.returncode != 0:
        log(f"push failed, will retry: {push.stderr.strip()}")
        return False
    log("pushed")
    return True


def unpushed(run):
    count = git("rev-list", "--count", f"origin/{run.branch}..HEAD")
    return count.returncode != 0 or count.stdout.strip() != "0"


def step1(run):
    tag = f"{run.label} STEP 1"
    if already_committed(tag) or not (run.trained("teacher") and run.trained("baseline")):
        return already_committed(tag)
    for name in ("teacher.log", "baseline.log", "pipeline.log"):
        strip_progress(run.logs / name)
    body = "\n".join([
        run.history_line("teacher", "Teacher FT (QLoRA nf4, GPU 0,1)"),
        run.history_line("baseline", "Student SFT (GPU 2,3)"),
        "",
        "KD 는 이어서 자동으로 시작된다. 로그는 진행 표시줄을 걸러냈다.",
    ])
    files = [run.logs / f for f in ("teacher_history.json", "baseline_history.json",
                                    "teacher.log", "baseline.log", "pipeline.log")]
    commit_and_push(run, files, f"results: {tag} — Teacher FT 와 Student SFT", body)
    return already_committed(tag)


def step2(run):
    tag = f"{run.label} STEP 2"
    if already_committed(tag) or not run.trained("distill"):
        return already_committed(tag)
    for name in ("distill.log", "pipeline.log"):
        strip_progress(run.logs / name)
    body = "\n".join([
        run.history_line("distill", f"KD ({run.kd_init}, Student GPU 0,1 / Teacher {run.teacher_precision} GPU 2,3)"),
        "",
        "평가, 비교, paired bootstrap 이 이어서 자동으로 실행된다.",
    ])
    files = [run.logs / f for f in ("distill_history.json", "distill.log", "pipeline.log")]
    commit_and_push(run, files, f"results: {tag} — KD 학습 이력", body)
    return already_committed(tag)


def step3(run):
    tag = f"{run.label} STEP 3"
    pipeline = run.logs / "pipeline.log"
    done = pipeline.is_file() and "PIPELINE_DONE" in pipeline.read_text(errors="replace")
    if already_committed(tag) or not done:
        return already_committed(tag)
    for path in run.logs.glob("*.log"):
        strip_progress(path)
    lines = []
    results = run.logs / "evaluation_results.json"
    if results.is_file():
        ppl = {r["name"]: r["perplexity"] for r in json.loads(results.read_text())}
        lines += [f"  {name:<16} {value:.3f}" for name, value in ppl.items()]
        if all(k in ppl for k in ("Teacher (FT)", "Student (FT)", "Student (KD)")):
            ft, kd, teacher = ppl["Student (FT)"], ppl["Student (KD)"], ppl["Teacher (FT)"]
            lines += ["", f"- FT -> KD {(kd - ft) / ft * 100:+.1f}%,"
                          f" 격차 중 KD 가 좁힌 비율 (FT-KD)/(FT-Teacher) = {(ft - kd) / (ft - teacher) * 100:.1f}%"]
    boot = run.logs / "paired_bootstrap.json"
    if boot.is_file():
        for c in json.loads(boot.read_text()).get("comparisons", []):
            if c.get("a") == "Student (KD)" and c.get("b") == "Student (FT)":
                lo, hi = c["ci95"]
                lines.append(f"- KD vs FT {c['observed']:+.3f}, paired bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}],"
                             f" KD 가 나은 chunk {c['chunks_a_better']}/{c['chunks']}")
    lines += ["", "자동 커밋이다. 해석과 문서는 별도 커밋으로 추가한다."]
    commit_and_push(run, [run.logs, run.figs], f"results: {tag} — {run.title} 4-way 평가", "\n".join(lines))
    return already_committed(tag)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--label", required=True, help='커밋 제목 접두어, 예: "Qwen3 v4"')
    parser.add_argument("--title", required=True, help='STEP 3 커밋 제목의 모델 설명, 예: "Qwen3 8B -> 1.7B"')
    args = parser.parse_args()
    run = Run(args.config, args.label, args.title)
    log(f"watcher started: {run.logs.relative_to(REPO)} on {run.branch}")
    while True:
        done1 = step1(run)
        done2 = done1 and step2(run)
        done3 = done2 and step3(run)
        if (done1 or done2 or done3) and unpushed(run):
            commit_and_push(run, [], "retry push", "")
        if done3 and not unpushed(run):
            log("all stages pushed; exiting")
            return
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
