#!/usr/bin/env python3
"""Figures for the six-combination KD summary (docs/small-model-kd-v1/MEETING_SUMMARY.md).

Color follows the study in every figure: blue = small-model-v1 (ours), orange = generic-kd-v1.
Palette validated with the dataviz validator (CVD dE 24.7, normal dE 33.6, light surface).
"""
import csv
import io
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'docs' / 'figures' / 'small-model-kd'
SURFACE, INK, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
OURS, THEIRS = '#2a78d6', '#eb6834'
LABEL = {'qwen3': 'Qwen3 8B→1.7B', 'gemma3': 'Gemma 3 12B→1B', 'kanana': 'Kanana 8B→2.1B',
         'qwen35': 'Qwen3.5 9B→4B', 'gemma4': 'Gemma 4 12B→E2B', 'lfm25': 'LFM2.5 24B→2.6B'}
STUDY = {'qwen3': OURS, 'gemma3': OURS, 'kanana': OURS, 'qwen35': THEIRS, 'gemma4': THEIRS, 'lfm25': THEIRS}
ORDER = ('kanana', 'qwen3', 'gemma3', 'qwen35', 'gemma4', 'lfm25')

for path in ('/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf',
             '/usr/share/fonts/truetype/nanum/NanumBarunGothicBold.ttf'):
    if Path(path).exists():
        font_manager.fontManager.addfont(path)
plt.rcParams.update({'font.family': 'NanumBarunGothic', 'axes.unicode_minus': False, 'font.size': 11,
                     'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
                     'text.color': INK, 'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': INK,
                     'axes.edgecolor': GRID})


def rows(text):
    return list(csv.DictReader(io.StringIO(text)))


def read(rel):
    return (REPO / rel).read_text()


def colleague(rel):
    return subprocess.run(['git', 'show', f'origin/feature/azure-h100-kd-tuning:{rel}'], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout


def style(ax, xlabel):
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel(xlabel)


def legend(fig):
    handles = [plt.Line2D([], [], marker='o', linestyle='', color=OURS, markersize=8, label='소형 모델 (small-model-v1)'),
               plt.Line2D([], [], marker='o', linestyle='', color=THEIRS, markersize=8, label='대형 모델 (generic-kd-v1)')]
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False, fontsize=10)


def fig_test_kd_effect():
    effects = rows(read('results/small-model-v1/paired_effects.csv'))
    effects += rows(colleague('results/published/generic-kd-v1/paired_effects.csv'))
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    for y, fam in enumerate(reversed(ORDER)):
        vals = [float(r['kd_relative_ppl_change_percent']) for r in effects if r['family'] == fam]
        ax.scatter(vals, [y] * len(vals), s=64, color=STUDY[fam], edgecolor=SURFACE, linewidth=2, zorder=3)
        mean = sum(vals) / len(vals)
        ax.plot([mean, mean], [y - 0.28, y + 0.28], color=INK, linewidth=2, zorder=4)
        ax.text(1.12, y, f'평균 {mean:+.2f}%', va='center', fontsize=10, color=INK)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(range(len(ORDER)), [LABEL[f] for f in reversed(ORDER)])
    ax.set_xlim(-1.6, 1.6)
    ax.set_xticks([-1.5, -1.0, -0.5, 0, 0.5, 1.0])
    style(ax, 'CE 대비 KD의 test PPL 변화 (%) · 왼쪽일수록 KD가 좋음 · 점 = seed 42·43·44, 막대 = 평균')
    legend(fig)
    ax.set_title('KD 효과 (같은 조합 안에서 CE와 비교, 내부 test)', loc='left', fontsize=13, color=INK, pad=12)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / 'fig1_kd_effect_test.png', dpi=160)


def fig_common_bpb():
    scores = rows(read('results/small-model-v1/common/scores.csv'))
    bpb = {}
    for r in scores:
        if r['model_id'].startswith(('ce-', 'kd-')):
            bpb.setdefault((r['family'], r['model_id'][:2]), []).append(float(r['bits_per_byte']))
    kd = {f: sum(bpb[f, 'kd']) / 3 for f in ORDER}
    ranked = sorted(ORDER, key=kd.get)
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    for y, fam in enumerate(reversed(ranked)):
        ax.barh(y, kd[fam], height=0.56, color=STUDY[fam], zorder=3)
        ax.text(kd[fam] + 0.008, y, f'{kd[fam]:.3f}', va='center', fontsize=10, color=INK)
    ax.set_yticks(range(len(ranked)), [f'{i}. {LABEL[f]}' for i, f in zip(range(6, 0, -1), reversed(ranked))])
    ax.set_xlim(0, 1.08)
    style(ax, 'KD Student의 bits-per-byte (seed 3개 평균) · 짧을수록 한국어 예측이 좋음')
    legend(fig)
    ax.set_title('6개 조합 공통 비교 (같은 validation 2,917,159바이트)', loc='left', fontsize=13, color=INK, pad=12)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / 'fig2_common_bpb.png', dpi=160)


def fig_retention():
    path = REPO / 'results/small-model-v1/retention/retention.csv'
    if not path.exists():
        print('retention results not available yet')
        return
    data = {(r['family'], r['unit']): r for r in rows(path.read_text())}
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), sharey=True)
    for ax, unit, title in ((axes[0], 'kobest', 'KoBEST 5개 과제 평균'), (axes[1], 'kmmlu', 'KMMLU 보건·사회복지·심리 평균')):
        for y, fam in enumerate(reversed(ORDER)):
            r = data[fam, unit]
            for key, marker, face in (('ce_vs_pre', 'o', SURFACE), ('kd_vs_pre', 'o', STUDY[fam])):
                off = 0.14 if key == 'ce_vs_pre' else -0.14
                x, lo, hi = float(r[f'{key}_pp']), float(r[f'{key}_lo']), float(r[f'{key}_hi'])
                ax.plot([lo, hi], [y + off] * 2, color=STUDY[fam], linewidth=2, zorder=2)
                ax.scatter([x], [y + off], s=60, marker=marker, facecolor=face, edgecolor=STUDY[fam], linewidth=2, zorder=3)
        ax.axvline(0, color=MUTED, linewidth=1)
        style(ax, '학습 전 대비 정답률 변화 (%p) · 선 = 95% 구간')
        ax.set_title(title, loc='left', fontsize=12, color=INK)
    axes[0].set_yticks(range(len(ORDER)), [LABEL[f] for f in reversed(ORDER)])
    handles = [plt.Line2D([], [], marker='o', linestyle='', markerfacecolor=SURFACE, markeredgecolor=MUTED,
                          markeredgewidth=2, markersize=8, label='CE (속이 빈 점)'),
               plt.Line2D([], [], marker='o', linestyle='', color=MUTED, markersize=8, label='KD (채운 점)'),
               plt.Line2D([], [], marker='s', linestyle='', color=OURS, markersize=8, label='소형 모델'),
               plt.Line2D([], [], marker='s', linestyle='', color=THEIRS, markersize=8, label='대형 모델')]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=10)
    fig.suptitle('기존 능력 유지: 추가 학습 뒤 한국어 이해 정답률 변화', x=0.01, ha='left', fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(OUT / 'fig3_retention.png', dpi=160)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_test_kd_effect()
    fig_common_bpb()
    fig_retention()
    print('figures written to', OUT.relative_to(REPO))


main()
