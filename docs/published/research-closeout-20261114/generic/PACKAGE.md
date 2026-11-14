# 기본 한국어 KD — 실험별 경량 재분석 자료

기존 [최종 24개 집계·9개 CE/KD 쌍](../../../../results/published/generic-kd-v1/README.md)과 조합하는 확장 기록이다. 최종 결과를 중복 복사하거나 재채점하지 않았다. [종료 감사](REPORT_KO.md)는 수치와 한계를 설명한다.

## 포함 파일

| 파일 | 내용 |
|---|---|
| `evidence.json` | 최종 24모델, 과거 14개 로그 summary, H100/초기 Qwen3.5 결과와 source SHA/pointer |
| `research-trials.json` | completed Student trial 76건: 공통 cohort 60 + 과거 pilot 16. 방법·LR·alpha·온도·seed·초기/최선 dev·종료/선택 step·정지 사유 |
| `research-projections.json` | 372개 JSON source의 설정·지표·history·선택·종료·안전한 오류 token projection. 중복 leaf-map 370개에 원래 origins를 연결 |
| `research-curves.json` | CSV 7개의 원래 순서·선택 열·216행. JSON history는 projections에 별도 보존 |
| `research-inventory.json` | `results/research/`의 1,385개 파일별 포함·제외·파싱 실패·기존 공개 중복 판정 |

76건은 독립 training seed 76개가 아니다. seed42의 여러 탐색과 서로 다른 cohort가 섞여 있으며, 계열별 Teacher는 하나로 고정했다. 원래 completed trial을 재개 횟수만큼 늘리지 않았다. `failure_tokens` 46건도 중복 가능한 문자열 occurrence이지 독립 실패 trial/incident 수가 아니다.

## 원본 연결을 읽는 법

- `leaves`의 키는 원본 JSON의 RFC6901 pointer다. `origins[].decode_chain`이 비어 있으면 원본 JSON 자체에 적용한다. 값이 있으면 `string_pointer`의 원래 문자열에서 `character_offset`부터 JSON을 decode하고 다음 체인을 순서대로 적용한 뒤 leaf pointer를 읽는다. 원래 stdout/stderr는 공개하지 않는다.
- `research-trials.json.source_sha256`은 **실제로 읽은 retained compact source 파일**의 SHA다. `historical_report_sha256`은 그 compact 안에 보존된 과거 원격 report의 SHA로, 둘은 다른 파일이다. `frozen_seed_source_sha256`은 별도 frozen-plan seed 근거의 파일 SHA다. 현재 source bytes와 과거 보고된 원격 payload를 혼동하지 않는다.
- 원본 metadata의 SHA와 공개 projection은 원본·checkpoint·Blob의 외부 서명이나 신규 전체 검증이 아니다. 공개 파일 SHA는 상위 [manifest](../MANIFEST.json)에 있다.

## 불완전한 범위를 숨기지 않기

Gemma/Liquid의 compact 보고는 전체 step 곡선이 아니라 partial snapshot이다. Qwen round1의 일부도 first/last snapshot이다. 없는 점은 보간하거나 합성하지 않았다. 출처 inventory의 제외 항목은 삭제하지 않고 이유를 남겼다. 개별 코퍼스 문서·생성문·prepared token arrays·가중치/optimizer·운영로그·full STATE·자격증명은 공개하지 않았다. inventory에서 SHA가 없는 제외 파일은 내용 미열람/미해시다. 해당 파일명이 있다고 공개 원문을 제공하는 것은 아니다.

따라서 이 묶음은 집계·곡선·선택/종료 및 설정을 재분석할 수 있게 하지만 문서군별 신뢰구간·전체 역사 실행·비공개 코퍼스 membership의 완전한 재현까지 제공하지 않는다. 기존 [평가·학습 helper](../../../FOUR_WAY_REPRODUCTION_GUIDE.md)는 별도로 유지한다.
