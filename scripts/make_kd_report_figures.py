"""3개 계열(Gemma 3 / Kanana 1.5 / Qwen3) v4 KD 결과를 모아 보고서용 지표와 차트를 만든다.

입력은 저장소에 커밋된 결과 파일뿐이다 (evaluation_results.json, paired_bootstrap.json, *_history.json,
results/qwen3/logs/qwen-family-bpb/bpb.json). 출력은 docs/figures/kd-midterm/ 아래의
밝은/어두운 테마 PNG 와 summary.json 이다.

bits-per-byte 추정: 세 계열의 test split 은 같은 500개 문서다 (문서 단위 split, seed 42).
각 계열의 test NLL 합(= 예측 토큰 수 × ln PPL)을 Qwen 측정에서 얻은 원문 바이트 수로 나눈다.
청크 첫 토큰과 꼬리 자투리만큼 바이트가 조금 다르므로 근사치다. Qwen3 행은 실측값과 대조한다.

  python3 scripts/make_kd_report_figures.py
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "figures" / "kd-midterm"

RUNS = {
    "Gemma 3": {"logs": "results/gemma3/logs/gemma3-12b-1b-v4", "pair": "12B → 1B", "lang": "다국어"},
    "Kanana 1.5": {"logs": "results/kanana/logs/kanana-8b-2b-v4", "pair": "8B → 2.1B", "lang": "한국어 특화"},
    "Qwen3": {"logs": "results/qwen3/logs/qwen3-8b-1.7b-v4", "pair": "8B → 1.7B", "lang": "다국어"},
}
BPB_REF = "results/qwen3/logs/qwen-family-bpb/bpb.json"

# 검증된 기본 categorical 팔레트의 앞 세 슬롯 (all-pairs 통과). 색은 계열을 따라가며 모든 차트에서 같다.
THEMES = {
    "light": {
        "surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7", "deemph": "#c3c2b7", "teacher": "#52514e",
        "series": {"Gemma 3": "#2a78d6", "Kanana 1.5": "#eb6834", "Qwen3": "#1baf7a"},
    },
    "dark": {
        "surface": "#1a1a19", "primary": "#f0efec", "secondary": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835", "deemph": "#52514e", "teacher": "#c3c2b7",
        "series": {"Gemma 3": "#3987e5", "Kanana 1.5": "#d95926", "Qwen3": "#199e70"},
    },
}
DPI = 200


def load(path):
    return json.loads((REPO / path).read_text())


def collect():
    ref = load(BPB_REF)
    ref_bytes = ref["models"]["Qwen3-1.7B Base"]["bytes"]
    rows = {}
    for family, run in RUNS.items():
        logs = run["logs"]
        evals = {r["name"]: r for r in load(f"{logs}/evaluation_results.json")}
        boot = load(f"{logs}/paired_bootstrap.json")
        kd_ft = next(c for c in boot["comparisons"] if c["a"] == "Student (KD)" and c["b"] == "Student (FT)")
        ppl = {k: evals[k]["perplexity"] for k in ("Teacher (FT)", "Student (KD)", "Student (FT)", "Student (Base)")}
        tokens = boot["tokens"]
        ft_boot = boot["perplexity"]["Student (FT)"]

        def bpb(p):
            return tokens * math.log(p) / (math.log(2) * ref_bytes)

        params = evals["Student (KD)"]["total_params"]
        rows[family] = {
            **run,
            "ppl": ppl,
            "teacher_params": evals["Teacher (FT)"]["total_params"],
            "student_params": params,
            "tokens_per_sec_student": evals["Student (KD)"]["tokens_per_sec"],
            "test_chunks": boot["chunks"],
            "test_tokens": tokens,
            "bytes_per_token": ref_bytes / tokens,
            "base_to_ft_pct": (ppl["Student (FT)"] - ppl["Student (Base)"]) / ppl["Student (Base)"] * 100,
            "ft_to_kd_pct": (ppl["Student (KD)"] - ppl["Student (FT)"]) / ppl["Student (FT)"] * 100,
            "kd_vs_ft": kd_ft["observed"],
            "kd_vs_ft_ci": kd_ft["ci95"],
            "kd_vs_ft_pct_ci": [c / ft_boot * 100 for c in kd_ft["ci95"]],
            "kd_vs_ft_pct_boot": kd_ft["observed"] / ft_boot * 100,
            "kd_better_chunks": kd_ft["chunks_a_better"] / kd_ft["chunks"] * 100,
            "gap_closed_pct": (ppl["Student (FT)"] - ppl["Student (KD)"])
                              / (ppl["Student (FT)"] - ppl["Teacher (FT)"]) * 100,
            "bpb_est": {k: bpb(v) for k, v in ppl.items()},
            "val_ce": {
                "teacher": [e["val_ce_loss"] for e in load(f"{logs}/teacher_history.json")],
                "sft": [e["val_ce_loss"] for e in load(f"{logs}/baseline_history.json")],
                "kd": [e["val_ce_loss"] for e in load(f"{logs}/distill_history.json")],
            },
            "mem_bf16_gb": params * 2 / 1e9,
            "mem_int4_gb": params * 4.5 / 8 / 1e9,
        }
    rows["_check_qwen3_bpb_measured"] = {
        "Student (Base)": ref["models"]["Qwen3-1.7B Base"]["bits_per_byte"],
        "Teacher (FT)": ref["models"]["Qwen3-8B Teacher FT"]["bits_per_byte"],
    }
    rows["_ref_bytes"] = ref_bytes
    return rows


# ---------- drawing helpers ----------

def setup_font():
    for path in ("/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
                 "/usr/share/fonts/truetype/nanum/NanumBarunGothicBold.ttf"):
        if Path(path).is_file():
            font_manager.fontManager.addfont(path)
    plt.rcParams["font.family"] = "NanumBarunGothic"
    plt.rcParams["axes.unicode_minus"] = False


def style_axes(ax, t, ygrid=True, xgrid=False):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(colors=t["muted"], labelcolor=t["secondary"], length=0, labelsize=9)
    if ygrid:
        ax.yaxis.grid(True, color=t["grid"], linewidth=0.8, linestyle="-")
    if xgrid:
        ax.xaxis.grid(True, color=t["grid"], linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)


def px_to_data(ax):
    """1 화면 픽셀이 x/y 데이터 단위로 얼마인지 (축 범위와 크기가 정해진 뒤 호출)."""
    bbox = ax.get_window_extent()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    return (x1 - x0) / bbox.width, (y1 - y0) / bbox.height


def rounded_bar(ax, x, height, width, color, radius_px=4):
    """끝은 4px 둥글게, 기준선 쪽은 각지게 그린 세로 막대."""
    xpp, ypp = px_to_data(ax)
    r = radius_px * xpp * (DPI / 100)
    aspect = ypp / xpp
    ax.add_patch(FancyBboxPatch((x - width / 2, 0), width, height,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                mutation_aspect=aspect, linewidth=0, facecolor=color, zorder=3))
    square = min(height, r * aspect * 1.05)
    ax.add_patch(Rectangle((x - width / 2, 0), width, square, linewidth=0, facecolor=color, zorder=3))


def figure(t, w, h, ncols=1, nrows=1, **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), dpi=DPI, **kw)
    fig.patch.set_facecolor(t["surface"])
    return fig, axes


def family_legend(fig, t, families, y=0.995):
    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=9, color=t["series"][f]) for f in families]
    labels = [f"{f} ({RUNS[f]['pair']})" for f in families]
    fig.legend(handles, labels, loc="upper center", ncol=len(families), frameon=False, bbox_to_anchor=(0.5, y),
               fontsize=9.5, labelcolor=t["secondary"], handletextpad=0.4, columnspacing=1.6)


def save(fig, name, theme):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}-{theme}.png", facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


# ---------- figures ----------

def fig_kd_gain(rows, theme):
    t = THEMES[theme]
    fams = list(RUNS)
    fig, (a1, a2) = figure(t, 9.6, 3.9, ncols=2)
    fig.subplots_adjust(wspace=0.32, top=0.80)
    family_legend(fig, t, fams)
    xs = range(len(fams))

    # (a) FT -> KD 개선율과 bootstrap 95% CI
    vals = [-rows[f]["kd_vs_ft_pct_boot"] for f in fams]
    cis = [[-c for c in rows[f]["kd_vs_ft_pct_ci"]] for f in fams]
    style_axes(a1, t)
    a1.set_xlim(-0.6, len(fams) - 0.4)
    a1.set_ylim(0, 7.6)
    a1.set_title("Student SFT 대비 KD의 PPL 개선율 (%)", color=t["primary"], fontsize=11, loc="left", pad=10)
    for x, f, v, (hi, lo) in zip(xs, fams, vals, cis):
        rounded_bar(a1, x, v, 0.26, t["series"][f])
        a1.plot([x, x], [lo, hi], color=t["secondary"], linewidth=1.2, zorder=4)
        for yy in (lo, hi):
            a1.plot([x - 0.06, x + 0.06], [yy, yy], color=t["secondary"], linewidth=1.2, zorder=4)
        a1.text(x + 0.17, v, f"{v:.1f}%", va="center", ha="left", color=t["primary"], fontsize=9.5)
    a1.set_xticks(list(xs), fams)
    a1.text(0, -0.2, "막대 = 평균 개선율, 세로선 = paired bootstrap 95% 신뢰구간 (높을수록 KD가 더 좋음)",
            transform=a1.transAxes, color=t["muted"], fontsize=8)

    # (b) Teacher-FT 격차 중 KD가 좁힌 비율
    vals = [rows[f]["gap_closed_pct"] for f in fams]
    style_axes(a2, t)
    a2.set_xlim(-0.6, len(fams) - 0.4)
    a2.set_ylim(0, 33)
    a2.set_title("Teacher와의 격차 중 KD가 좁힌 비율 (%)", color=t["primary"], fontsize=11, loc="left", pad=10)
    for x, f, v in zip(xs, fams, vals):
        rounded_bar(a2, x, v, 0.26, t["series"][f])
        a2.text(x, v + 0.6, f"{v:.1f}%", va="bottom", ha="center", color=t["primary"], fontsize=9.5)
    a2.set_xticks(list(xs), fams)
    a2.text(0, -0.2, "(FT - KD) / (FT - Teacher). tokenizer와 무관한 상대 지표",
            transform=a2.transAxes, color=t["muted"], fontsize=8)
    save(fig, "kd-gain", theme)


def fig_four_way(rows, theme):
    t = THEMES[theme]
    fams = list(RUNS)
    roles = ["Teacher (FT)", "Student (KD)", "Student (FT)", "Student (Base)"]
    short = ["Teacher\nFT", "Student\nKD", "Student\nSFT", "Student\nBase"]
    fig, axes = figure(t, 10.4, 3.9, ncols=3)
    fig.subplots_adjust(wspace=0.28, top=0.78)
    for ax, f in zip(axes, fams):
        style_axes(ax, t)
        ppl = rows[f]["ppl"]
        ymax = max(ppl.values()) * 1.18
        ax.set_xlim(-0.6, len(roles) - 0.4)
        ax.set_ylim(0, ymax)
        ax.set_title(f"{f}  {RUNS[f]['pair']}", color=t["primary"], fontsize=11, loc="left", pad=10)
        for x, role in enumerate(roles):
            color = t["series"][f] if role == "Student (KD)" else (t["teacher"] if role == "Teacher (FT)" else t["deemph"])
            rounded_bar(ax, x, ppl[role], 0.34, color)
            weight = "bold" if role == "Student (KD)" else "normal"
            ax.text(x, ppl[role] + ymax * 0.015, f"{ppl[role]:.2f}", ha="center", va="bottom",
                    color=t["primary"], fontsize=9, fontweight=weight)
        ax.set_xticks(range(len(roles)), short)
    fig.text(0.125, 0.955, "test PPL (낮을수록 좋음) — 계열마다 tokenizer가 달라 PPL은 같은 계열 안에서만 비교",
             color=t["secondary"], fontsize=9.5)
    fig.text(0.125, 0.905, "색칠한 막대 = KD Student, 진회색 = Teacher (FT), 연회색 = 비교 대상 Student",
             color=t["muted"], fontsize=8.5)
    save(fig, "four-way-ppl", theme)


def fig_val_ce(rows, theme):
    t = THEMES[theme]
    fams = list(RUNS)
    fig, axes = figure(t, 10.4, 3.7, ncols=3)
    fig.subplots_adjust(wspace=0.3, top=0.78)
    for ax, f in zip(axes, fams):
        style_axes(ax, t)
        sft, kd = rows[f]["val_ce"]["sft"], rows[f]["val_ce"]["kd"]
        ax.plot(range(1, len(sft) + 1), sft, color=t["deemph"], linewidth=2, marker="o", markersize=5,
                markeredgecolor=t["surface"], markeredgewidth=1.5, solid_capstyle="round", zorder=3)
        ax.plot(range(1, len(kd) + 1), kd, color=t["series"][f], linewidth=2, marker="o", markersize=5,
                markeredgecolor=t["surface"], markeredgewidth=1.5, solid_capstyle="round", zorder=4)
        ax.text(len(sft) + 0.12, sft[-1], "SFT", color=t["secondary"], fontsize=9, va="center")
        ax.text(len(kd) + 0.12, kd[-1], "KD", color=t["primary"], fontsize=9, va="center", fontweight="bold")
        lo, hi = min(sft + kd), max(sft + kd)
        pad = (hi - lo) * 0.25
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(0.7, 4.6)
        ax.set_xticks(range(1, 5), [f"epoch {i}" for i in range(1, 5)])
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02 if hi - lo < 0.2 else 0.05))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
        ax.set_title(f"{f}  {RUNS[f]['pair']}", color=t["primary"], fontsize=11, loc="left", pad=10)
    fig.text(0.125, 0.955, "validation CE (낮을수록 좋음) — SFT는 1~2 epoch 뒤 과적합, KD는 더 오래 개선",
             color=t["secondary"], fontsize=9.5)
    fig.text(0.125, 0.905, "회색 = Student SFT (CE만), 색 = Student KD. 세로축은 계열마다 다르다",
             color=t["muted"], fontsize=8.5)
    save(fig, "val-ce-curves", theme)


def fig_edge(rows, theme):
    t = THEMES[theme]
    fams = list(RUNS)
    fig, ax = figure(t, 7.6, 4.6)
    fig.subplots_adjust(top=0.80)
    style_axes(ax, t, ygrid=True, xgrid=True)
    ax.spines["bottom"].set_visible(False)
    family_legend(fig, t, fams, y=1.0)
    for f in fams:
        r = rows[f]
        x = r["student_params"] / 1e9
        base, kd = r["bpb_est"]["Student (Base)"], r["bpb_est"]["Student (KD)"]
        c = t["series"][f]
        ax.annotate("", xy=(x, kd + 0.006), xytext=(x, base - 0.006),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.6, shrinkA=0, shrinkB=0), zorder=3)
        ax.scatter([x], [base], s=60, facecolors=t["surface"], edgecolors=c, linewidths=2, zorder=4)
        ax.scatter([x], [kd], s=80, color=c, edgecolors=t["surface"], linewidths=2, zorder=5)
        ax.text(x + 0.05, kd, f"KD {kd:.3f}", color=t["primary"], fontsize=9.5, va="center", fontweight="bold")
        ax.text(x + 0.05, base, f"Base {base:.3f}", color=t["secondary"], fontsize=9, va="center")
    ax.set_xlim(0.7, 2.6)
    ax.set_ylim(0.78, 1.18)
    ax.set_xlabel("Student 파라미터 수 (B)", color=t["secondary"], fontsize=9.5)
    ax.set_ylabel("bits-per-byte (추정, 낮을수록 좋음)", color=t["secondary"], fontsize=9.5)
    fig.text(0.125, 0.875, "엣지 후보 Student: 크기 대비 한국어 예측력 — 같은 test 원문 기준, 왼쪽 아래일수록 유리",
             color=t["secondary"], fontsize=9.5)
    fig.text(0.125, 0.835, "빈 원 = 학습 전 Base, 채운 원 = KD 후. bits-per-byte는 PPL과 원문 바이트 수로 환산한 근사치",
             color=t["muted"], fontsize=8.5)
    save(fig, "edge-tradeoff", theme)


def main():
    rows = collect()
    setup_font()
    for theme in THEMES:
        fig_kd_gain(rows, theme)
        fig_four_way(rows, theme)
        fig_val_ce(rows, theme)
        fig_edge(rows, theme)
    (OUT / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    for f in RUNS:
        r = rows[f]
        print(f"{f:<11} PPL T/KD/FT/Base " + " / ".join(f"{v:.3f}" for v in r["ppl"].values())
              + f" | Base->FT {r['base_to_ft_pct']:+.1f}% | FT->KD {r['ft_to_kd_pct']:+.2f}%"
              + f" (boot {r['kd_vs_ft_pct_boot']:+.2f}%, CI {r['kd_vs_ft_pct_ci'][0]:+.2f}..{r['kd_vs_ft_pct_ci'][1]:+.2f}%)"
              + f" | better {r['kd_better_chunks']:.1f}% | gap {r['gap_closed_pct']:.1f}%")
        print(f"{'':<11} bpb_est " + " / ".join(f"{k} {v:.4f}" for k, v in r["bpb_est"].items())
              + f" | bytes/token {r['bytes_per_token']:.3f} | params {r['student_params']/1e9:.3f}B"
              + f" (teacher {r['teacher_params']/1e9:.2f}B) | mem bf16 {r['mem_bf16_gb']:.2f}GB int4 {r['mem_int4_gb']:.2f}GB"
              + f" | tok/s {r['tokens_per_sec_student']:.0f} | chunks {r['test_chunks']} tokens {r['test_tokens']}")
        print(f"{'':<11} val CE teacher {r['val_ce']['teacher']} sft {r['val_ce']['sft']} kd {r['val_ce']['kd']}")
    print("Qwen3 bpb measured (check):", rows["_check_qwen3_bpb_measured"], "ref bytes", rows["_ref_bytes"])
    print("saved:", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
