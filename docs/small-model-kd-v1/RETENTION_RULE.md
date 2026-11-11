# 기존 능력 유지 평가 규칙

2026-11-12, 결과를 보기 전에 고정했습니다. 동작 확인은 문항 일부로만 했고, 그 값은 비교에 쓰지 않습니다.

## 질문

한국어 일반 문서로 추가 학습(CE·KD)한 Student가 원래의 한국어 이해 능력을 유지하는가? KD가 CE보다 더 잘 유지하는가?

## 대상과 데이터

- 모델: 6개 조합 × (학습 전 Student, CE seed 42·43·44, KD seed 42·43·44) = 42개. 공통 채점과 같은 모델입니다.
- KoBEST(skt/kobest_v1, CC BY-SA 4.0) test: BoolQ 1,404, COPA 1,000, HellaSwag 500, SentiNeg 397, WiC 1,260문항.
- KMMLU(HAERAE-HUB/KMMLU, CC BY-ND 4.0) test: 보건(Health) 100, 사회복지(Social-Welfare) 1,000, 심리(Psychology) 1,000문항.
- 문항 원문은 커밋하지 않습니다. 문항 파일 SHA-256은 `results/small-model-v1/retention/items-summary.json`에 있습니다.

## 채점

- 0-shot. 보기마다 문맥 뒤에 오는 보기 토큰의 log-probability 합을 계산합니다. BOS 없음, 문맥은 왼쪽을 잘라 1,024토큰 이내로 맞춥니다.
- KoBEST 프롬프트는 lm-evaluation-harness의 KoBEST 형식을 따르고, KMMLU는 `질문 / A~D 보기 / 정답:` 뒤에 ` A`~` D`를 채점합니다.
- 지표: COPA·HellaSwag는 보기 바이트 길이로 나눈 값(acc_norm), 나머지는 합(acc)으로 고른 정답률.
- 묶음 점수: KoBEST 5개 과제 평균, KMMLU 3개 과목 평균(과목별 단순 평균).

## 판정

- 조합마다 학습 전, CE 평균(seed 3개), KD 평균(seed 3개)의 정답률을 보고합니다.
- 차이 세 가지: CE − 학습 전, KD − 학습 전, KD − CE(%p).
- 불확실성: 과제별로 문항을 복원추출해 10,000번 다시 계산한 95% 구간.
- 구간이 0보다 크면 향상, 0보다 작으면 하락, 0을 포함하면 변화 없음.

## 한계

- 학습 전 모델도 BOS 없이 채점합니다. 학습과 같은 조건이지만, 대화용 모델의 원래 성능보다 낮게 나올 수 있습니다.
- 0-shot 확률 비교는 생성 품질이나 대화 능력을 직접 재지 않습니다.
- 구간은 문항 표본의 불확실성만 반영합니다. seed 3개의 차이는 평균에 포함됩니다.
- 6개 조합 × 3가지 비교에 다중 비교 보정은 하지 않습니다.
