# 실험 노트 — gemma3-12b-1b-v3-fullvocab

## 자동 진단

- ⚠️ KD(11.70) >= FT(11.66) — 증류 효과 미확인

## 다음 액션

- [ ] temperature grid 탐색 (예: 1, 2, 4)
- [ ] alpha 값 높이기 (CE 비율 늘리기)

## 메모 (직접 작성)

<!-- 여기에 실험 해석, 인사이트, 결정사항 등을 기록하세요 -->

- full-vocab KD 11.699 vs FT 11.663, paired bootstrap 95% CI [−0.057, +0.144] — 노이즈와 구분되지 않는다. top-K 128과의 차이도 CI [−0.156, +0.040]로 유의하지 않다 (`../gemma3-12b-1b-v3/paired_bootstrap.json`).
- 해석 전체: [docs/gemma3-4way-topk-kd.md](../../../../docs/gemma3-4way-topk-kd.md)


