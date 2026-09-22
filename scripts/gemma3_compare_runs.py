"""top-K KD 와 full-vocab KD 실행을 나란히 비교한다.

두 실행은 Teacher FT 어댑터와 KD 초기 Student SFT 를 공유하고 kd_top_k 만 다르다.
따라서 Student (KD) 행의 차이가 곧 목표 분포를 자른 대가다.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TOPK_RUN = "gemma3-12b-1b-v3"
FULL_RUN = "gemma3-12b-1b-v3-fullvocab"
LOGS = Path("results/gemma3/logs")


def load(run_id, name):
    path = LOGS / run_id / name
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def by_name(results):
    return {row["name"]: row for row in results} if results else {}


def main() -> int:
    topk = by_name(load(TOPK_RUN, "evaluation_results.json"))
    full = by_name(load(FULL_RUN, "evaluation_results.json"))
    if not topk:
        print(f"⚠️ top-K 실행 결과가 없습니다: {LOGS / TOPK_RUN}")
        return 1

    print("\n" + "=" * 78)
    print("4-Way 결과 — top-K(128) KD")
    print("=" * 78)
    print(f"{'모델':<18}{'파라미터':>16}{'PPL':>10}{'tokens/s':>11}{'ms/tok':>9}")
    print("-" * 78)
    order = ["Teacher (FT)", "Teacher", "Student (KD)", "Student (FT)", "Student (Base)"]
    for name in order:
        row = topk.get(name)
        if row:
            print(f"{name:<18}{row['total_params']:>16,}{row['perplexity']:>10.3f}"
                  f"{row['tokens_per_sec']:>11.0f}{row['ms_per_token']:>9.3f}")

    if full:
        print("\n" + "=" * 78)
        print("KD 방식 A/B — Teacher FT 어댑터와 Student SFT 초기값을 공유, kd_top_k 만 다름")
        print("=" * 78)
        print(f"{'':<22}{'top-K(128)':>14}{'full-vocab':>14}{'차이':>12}")
        print("-" * 78)
        a, b = topk.get("Student (KD)"), full.get("Student (KD)")
        if a and b:
            for label, key, fmt in [
                ("Student (KD) PPL", "perplexity", "{:.3f}"),
                ("tokens/s", "tokens_per_sec", "{:.0f}"),
            ]:
                diff = b[key] - a[key]
                print(f"{label:<22}{fmt.format(a[key]):>14}{fmt.format(b[key]):>14}"
                      f"{diff:>+12.3f}")

        ft = topk.get("Student (FT)")
        if ft:
            print()
            print(f"  기준선 Student (FT) PPL: {ft['perplexity']:.3f}")
            for label, row in [("top-K(128)", a), ("full-vocab", b)]:
                if row:
                    gain = ft["perplexity"] - row["perplexity"]
                    pct = gain / ft["perplexity"] * 100
                    verdict = "증류 효과 있음" if gain > 0 else "증류 효과 없음"
                    print(f"    {label:<12} KD {row['perplexity']:.3f} → "
                          f"FT 대비 {gain:+.3f} ({pct:+.1f}%) — {verdict}")
    else:
        print(f"\n⚠️ full-vocab 실행 결과가 아직 없습니다: {LOGS / FULL_RUN}")

    ablation = load(TOPK_RUN, "topk_ablation.json")
    if ablation:
        print("\n" + "=" * 78)
        print(f"정적 측정 — 같은 Student 상태, {ablation['batches']}개 배치")
        print("=" * 78)
        print(f"  KD 손실          : full {ablation['mean_kd_full']:.4f} → "
              f"top-K {ablation['mean_kd_topk']:.4f}")
        print(f"  CE 손실          : {ablation['mean_ce_full']:.4f} (top-K 와 무관해야 정상)")
        print(f"  gradient 코사인  : {ablation['mean_grad_cosine']:.6f}")
        print(f"  gradient norm 비 : {ablation['mean_grad_norm_ratio']:.4f}")
        print(f"  |Δg|/|g|         : {ablation['mean_grad_rel_diff']:.4f}")
        print("\n  코사인이 1 에 가까울수록 top-K 는 같은 방향으로 더 싸게 가는 근사다.")
        print("  멀어질수록 다른 목적함수를 최적화하는 것이므로 Qwen v3 의 KD 와")
        print("  같은 축에서 비교하면 안 된다.")

    # 학습 곡선 요약
    print("\n" + "=" * 78)
    print("KD 학습 곡선 (val CE)")
    print("=" * 78)
    for label, run_id in [("top-K(128)", TOPK_RUN), ("full-vocab", FULL_RUN)]:
        history = load(run_id, "distill_history.json")
        if not history:
            print(f"  {label:<12} (없음)")
            continue
        values = [f"{h['val_ce_loss']:.4f}" for h in history]
        best = min(h["val_ce_loss"] for h in history)
        print(f"  {label:<12} best {best:.4f} | {' → '.join(values)}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
