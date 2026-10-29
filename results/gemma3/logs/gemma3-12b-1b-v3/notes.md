# 실험 노트 — gemma3-12b-1b-v3

## 자동 진단

- ✅ KD(11.64) < FT(11.66) — 증류 효과 확인됨

## 다음 액션


## 메모 (직접 작성)

<!-- 여기에 실험 해석, 인사이트, 결정사항 등을 기록하세요 -->

- 위 자동 진단은 KD < FT이면 무조건 효과 있음으로 적는다. KD−FT 차이 0.02는 paired bootstrap 95% CI [−0.148, +0.118]이 0을 포함해 **노이즈와 구분되지 않는다** (`paired_bootstrap.json`).
- 해석 전체: [docs/gemma3-4way-topk-kd.md](../../../../docs/gemma3-4way-topk-kd.md)


