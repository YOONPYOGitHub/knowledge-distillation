"""4-Way 비교 & 시각화"""

import json
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np

from src.config import KDConfig


def load_results(config: KDConfig) -> list:
    """저장된 평가 결과 로드"""
    results_path = config.log_dir / "evaluation_results.json"
    with open(results_path) as f:
        return json.load(f)


def print_comparison_table(results: list):
    """4-Way 비교 테이블 출력"""
    print("\n" + "=" * 70)
    print("4-Way 비교 결과")
    print("=" * 70)
    print(f"{'모델':<18} {'파라미터':>12} {'PPL':>10} {'tokens/s':>10} {'ms/tok':>8}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['name']:<18} "
            f"{r['total_params']:>12,} "
            f"{r['perplexity']:>10.2f} "
            f"{r['tokens_per_sec']:>10.0f} "
            f"{r['ms_per_token']:>8.2f}"
        )
    print("=" * 70)

    # 핵심 비교
    names = {r["name"]: r for r in results}
    if "Student (KD)" in names and "Student (FT)" in names:
        kd_ppl = names["Student (KD)"]["perplexity"]
        ft_ppl = names["Student (FT)"]["perplexity"]
        diff = ft_ppl - kd_ppl
        pct = (diff / ft_ppl) * 100
        print(f"\n📊 증류 효과 (Q2): KD PPL {kd_ppl:.2f} vs FT PPL {ft_ppl:.2f}")
        if diff > 0:
            print(f"   → KD가 FT 대비 PPL {diff:.2f} 낮음 ({pct:.1f}% 개선)")
        else:
            print(f"   → FT가 KD 대비 PPL {-diff:.2f} 낮음")


def plot_perplexity(results: list, save_path: str):
    """Perplexity 비교 막대 그래프"""
    names = [r["name"] for r in results]
    ppls = [r["perplexity"] for r in results]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9E9E9E"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, ppls, color=colors[: len(names)], edgecolor="black", linewidth=0.5)

    for bar, ppl in zip(bars, ppls):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{ppl:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Perplexity (↓ lower is better)")
    ax.set_title("4-Way Perplexity Comparison")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"차트 저장 → {save_path}")


def plot_speed(results: list, save_path: str):
    """추론 속도 비교 막대 그래프"""
    names = [r["name"] for r in results]
    speeds = [r["tokens_per_sec"] for r in results]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9E9E9E"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, speeds, color=colors[: len(names)], edgecolor="black", linewidth=0.5)

    for bar, spd in zip(bars, speeds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                f"{spd:.0f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Tokens/sec (↑ higher is better)")
    ax.set_title("4-Way Inference Speed Comparison")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"차트 저장 → {save_path}")


def plot_training_curves(config: KDConfig, save_path: str):
    """Teacher FT / KD / FT 학습 Loss 곡선 비교"""
    kd_path = config.log_dir / "distill_history.json"
    ft_path = config.log_dir / "baseline_history.json"
    teacher_path = config.log_dir / "teacher_history.json"

    if not kd_path.exists() or not ft_path.exists():
        print("⚠️ 학습 로그가 부족하여 학습 곡선을 그릴 수 없습니다.")
        return

    with open(kd_path) as f:
        kd_hist = json.load(f)
    with open(ft_path) as f:
        ft_hist = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Teacher FT 곡선 (있을 때만)
    if teacher_path.exists():
        with open(teacher_path) as f:
            teacher_hist = json.load(f)
        t_epochs = [h["epoch"] for h in teacher_hist]
        t_ce = [h["ce_loss"] for h in teacher_hist]
        ax.plot(t_epochs, t_ce, "D-", label="Teacher (FT) - CE Loss", color="#2196F3")

    kd_epochs = [h["epoch"] for h in kd_hist]
    kd_ce = [h["ce_loss"] for h in kd_hist]
    ft_epochs = [h["epoch"] for h in ft_hist]
    ft_ce = [h["ce_loss"] for h in ft_hist]

    ax.plot(kd_epochs, kd_ce, "o-", label="Student (KD) - CE Loss", color="#4CAF50")
    ax.plot(ft_epochs, ft_ce, "s-", label="Student (FT) - CE Loss", color="#FF9800")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("CE Loss")
    ax.set_title("Training Loss: Teacher FT / KD / FT")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"차트 저장 → {save_path}")


def plot_val_loss_curves(config: KDConfig, save_path: str):
    """KD / FT Validation CE Loss 곡선 비교 (역전 포인트 강조)"""
    kd_path = config.log_dir / "distill_history.json"
    ft_path = config.log_dir / "baseline_history.json"

    if not kd_path.exists() or not ft_path.exists():
        print("⚠️ 학습 로그가 부족하여 Val Loss 곡선을 그릴 수 없습니다.")
        return

    with open(kd_path) as f:
        kd_hist = json.load(f)
    with open(ft_path) as f:
        ft_hist = json.load(f)

    kd_epochs = [h["epoch"] for h in kd_hist]
    kd_val = [h["val_ce_loss"] for h in kd_hist]
    ft_epochs = [h["epoch"] for h in ft_hist]
    ft_val = [h["val_ce_loss"] for h in ft_hist]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(kd_epochs, kd_val, "o-", label="Student (KD) - Val CE", color="#4CAF50", linewidth=2)
    ax.plot(ft_epochs, ft_val, "s-", label="Student (FT) - Val CE", color="#FF9800", linewidth=2)

    # FT best 수평선
    ft_best = min(ft_val)
    ax.axhline(y=ft_best, color="#FF9800", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(max(kd_epochs) + 0.1, ft_best, f"FT best={ft_best:.3f}",
            fontsize=8, color="#FF9800", va="center")

    # 역전 포인트 탐색 및 표시
    kd_best = min(kd_val)
    if kd_best < ft_best:
        # KD가 FT best를 처음 이긴 epoch 찾기
        for i, (ep, val) in enumerate(zip(kd_epochs, kd_val)):
            if val < ft_best:
                ax.annotate(f"KD < FT best\n(ep{ep})",
                            xy=(ep, val), xytext=(ep + 0.5, val + 0.02),
                            fontsize=9, color="#4CAF50", fontweight="bold",
                            arrowprops=dict(arrowstyle="->", color="#4CAF50"))
                break

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation CE Loss (↓ lower is better)")
    ax.set_title("Validation Loss: KD vs FT (Overfitting Contrast)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"차트 저장 → {save_path}")


def generate_summary(results: list, config: KDConfig) -> dict:
    """자동 진단 포함 summary.json 생성"""
    names = {r["name"]: r for r in results}
    diagnosis = []
    next_actions = []

    # PPL 이상치 진단
    for r in results:
        if r["perplexity"] > 10000:
            diagnosis.append(
                f"⚠️ {r['name']} PPL={r['perplexity']:.0f} — "
                f"비정상적으로 높음 (패딩 과다 또는 미학습)"
            )
        if r["perplexity"] < 1.5 and r["name"] != "Teacher":
            diagnosis.append(
                f"⚠️ {r['name']} PPL={r['perplexity']:.2f} — "
                f"과적합 의심 (패딩 토큰 예측으로 loss 왜곡 가능)"
            )

    # KD vs FT 비교
    if "Student (KD)" in names and "Student (FT)" in names:
        kd_ppl = names["Student (KD)"]["perplexity"]
        ft_ppl = names["Student (FT)"]["perplexity"]
        if kd_ppl < ft_ppl:
            diagnosis.append(
                f"✅ KD({kd_ppl:.2f}) < FT({ft_ppl:.2f}) — 증류 효과 확인됨"
            )
        else:
            diagnosis.append(
                f"⚠️ KD({kd_ppl:.2f}) >= FT({ft_ppl:.2f}) — 증류 효과 미확인"
            )
            next_actions.append("temperature 값 조정 (현재 → +2 시도)")
            next_actions.append("alpha 값 낮추기 (CE 비율 줄이기)")

    # epoch 부족 진단
    if config.epochs <= 1:
        diagnosis.append("⚠️ 1 epoch만 실행 — 수렴 전 평가일 수 있음")
        next_actions.append("epochs를 3~5로 늘려서 수렴 확인")

    # seq_length 진단
    if config.max_seq_length <= 128:
        diagnosis.append(
            f"⚠️ max_seq_length={config.max_seq_length} — 패딩 비율 높을 수 있음"
        )
        next_actions.append("max_seq_length를 512로 올려서 패딩 비율 줄이기")

    # 서버 실험 권장
    if config.device != "cuda":
        next_actions.append("GPU 서버에서 server_config()로 본 실험 수행")

    summary = {
        "run_id": config.run_id,
        "config": {
            "teacher_model": config.teacher_model,
            "student_model": config.student_model,
            "temperature": config.temperature,
            "alpha": config.alpha,
            "epochs": config.epochs,
            "teacher_epochs": config.teacher_epochs,
            "teacher_learning_rate": config.teacher_learning_rate,
            "teacher_checkpoint": config.teacher_checkpoint,
            "batch_size": config.batch_size,
            "max_seq_length": config.max_seq_length,
            "device": config.device,
        },
        "results": {r["name"]: {"ppl": r["perplexity"], "tokens_per_sec": r["tokens_per_sec"]} for r in results},
        "diagnosis": diagnosis,
        "next_actions": next_actions,
    }

    summary_path = config.log_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"서머리 저장 → {summary_path}")

    # notes.md 템플릿 생성 (없을 때만)
    notes_path = config.log_dir / "notes.md"
    if not notes_path.exists():
        notes_content = f"# 실험 노트 — {config.run_id}\n\n"
        notes_content += "## 자동 진단\n\n"
        for d in diagnosis:
            notes_content += f"- {d}\n"
        notes_content += "\n## 다음 액션\n\n"
        for a in next_actions:
            notes_content += f"- [ ] {a}\n"
        notes_content += "\n## 메모 (직접 작성)\n\n"
        notes_content += "<!-- 여기에 실험 해석, 인사이트, 결정사항 등을 기록하세요 -->\n\n"
        with open(notes_path, "w", encoding="utf-8") as f:
            f.write(notes_content)
        print(f"노트 템플릿 생성 → {notes_path}")

    return summary


def compare(config: KDConfig):
    """4-Way 비교 전체 실행"""
    print("=== 4-Way Comparison ===\n")
    config.ensure_dirs()

    results = load_results(config)
    print_comparison_table(results)

    fig_dir = config.figure_dir
    plot_perplexity(results, str(fig_dir / "perplexity_comparison.png"))
    plot_speed(results, str(fig_dir / "inference_speed.png"))
    plot_training_curves(config, str(fig_dir / "training_loss.png"))
    plot_val_loss_curves(config, str(fig_dir / "val_loss.png"))

    summary = generate_summary(results, config)

    print("\n--- 자동 진단 ---")
    for d in summary["diagnosis"]:
        print(f"  {d}")
    if summary["next_actions"]:
        print("\n--- 다음 액션 ---")
        for a in summary["next_actions"]:
            print(f"  → {a}")

    print("\n모든 비교 완료!")


if __name__ == "__main__":
    from src.config import local_config

    config = local_config()
    compare(config)
