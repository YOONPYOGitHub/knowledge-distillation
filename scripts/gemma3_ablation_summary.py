"""v3 요인 분리 ablation(초기값 × divergence) 결과를 2×2 표로 모은다.

네 실행은 Teacher FT 어댑터, 데이터, T/α/lr/top-K 를 공유하고
KD 초기값(Student SFT / pretrained)과 divergence(reverse / forward KL)만 다르다.

결과: results/gemma3/logs/gemma3-12b-1b-v3/ablation_summary.json
PPL 차이의 유의성은 scripts/gemma3_paired_bootstrap.py ... ablation 으로 따로 판정한다.
"""

import json
from pathlib import Path

LOGS = Path("results/gemma3/logs")

# (초기값, divergence) -> run_id
CELLS = {
    ("FT", "reverse_kl"): "gemma3-12b-1b-v3",
    ("FT", "forward_kl"): "gemma3-12b-1b-v3-abl-ft-fwd",
    ("Base", "reverse_kl"): "gemma3-12b-1b-v3-abl-base-rev",
    ("Base", "forward_kl"): "gemma3-12b-1b-v3-abl-base-fwd",
}


def load(run_id, name):
    path = LOGS / run_id / name
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def ppl_of(run_id, model):
    rows = load(run_id, "evaluation_results.json") or []
    for row in rows:
        if row["name"] == model:
            return row["perplexity"]
    return None


def main() -> int:
    reference = CELLS[("FT", "reverse_kl")]
    ft = ppl_of(reference, "Student (FT)")
    base = ppl_of(reference, "Student (Base)")

    cells = []
    for (init, divergence), run_id in CELLS.items():
        history = load(run_id, "distill_history.json")
        kd = ppl_of(run_id, "Student (KD)")
        cell = {"init": init, "divergence": divergence, "run_id": run_id, "kd_ppl": kd}
        if history:
            best = min(history, key=lambda h: h["val_ce_loss"])
            cell.update({
                "epochs_run": len(history),
                "best_epoch": best["epoch"],
                "best_val_ce": best["val_ce_loss"],
                "val_ce_curve": [h["val_ce_loss"] for h in history],
                "minutes": sum(h["time"] for h in history) / 60,
            })
        if kd is not None and ft is not None:
            cell["vs_ft"] = kd - ft
        cells.append(cell)

    print("\n" + "=" * 78)
    print("v3 요인 분리 — Student (KD) PPL  (공통: T=2, α=0.5, lr 1e-5, tokenmean, top-K 128)")
    print("=" * 78)
    print(f"  Student (FT) {ft:.3f} | Student (Base) {base:.3f}" if ft and base else "  기준선 없음")
    print(f"\n{'':<14}{'reverse KL':>16}{'forward KL':>16}")
    for init in ["FT", "Base"]:
        row = [c for c in cells if c["init"] == init]
        values = []
        for divergence in ["reverse_kl", "forward_kl"]:
            cell = next(c for c in row if c["divergence"] == divergence)
            values.append(f"{cell['kd_ppl']:.3f}" if cell["kd_ppl"] is not None else "—")
        print(f"{init + ' 초기값':<14}{values[0]:>16}{values[1]:>16}")

    print(f"\n{'실행':<34}{'KD PPL':>9}{'FT 대비':>9}{'best ep':>9}{'best vCE':>10}{'epochs':>8}{'분':>6}")
    print("-" * 85)
    for cell in cells:
        if "best_epoch" not in cell:
            print(f"{cell['run_id']:<34}  (결과 없음)")
            continue
        kd = f"{cell['kd_ppl']:.3f}" if cell["kd_ppl"] is not None else "—"
        vs = f"{cell['vs_ft']:+.3f}" if "vs_ft" in cell else "—"
        print(f"{cell['run_id']:<34}{kd:>9}{vs:>9}{cell['best_epoch']:>9}"
              f"{cell['best_val_ce']:>10.4f}{cell['epochs_run']:>8}{cell['minutes']:>6.0f}")

    print("\nval CE 곡선")
    for cell in cells:
        if "val_ce_curve" in cell:
            curve = " → ".join(f"{v:.4f}" for v in cell["val_ce_curve"])
            print(f"  {cell['init']:>4}+{cell['divergence'][:3]}  {curve}")

    out = LOGS / reference / "ablation_summary.json"
    with open(out, "w", encoding="utf-8") as handle:
        json.dump({"student_ft_ppl": ft, "student_base_ppl": base, "cells": cells},
                  handle, ensure_ascii=False, indent=2)
    print(f"\n저장 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
