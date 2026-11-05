# Common Korean v1 노출 이력 판정 — 2026-10-26 17:44 KST

상태: **제한된 증거 검토 완료. 완전한 역사적 비접근 증명 아님. 최종평가 실행 gate는 별도 미완료.**

## 확인한 사실
- 공통 준비 source `common-corpus-v1-reviewed`와 Teacher retry1, Student round1 retry1의 `src/research_data.py` SHA-256은 모두 `3f4e28fd77b93667467fc8697c86eaaee24c1ac0e33fd72f030c21cbb0422696`이다.
- Student round1 retry1 plan의 source 10개 해시가 원격 고정 snapshot과 일치했다. Teacher run에는 같은 경로의 plan.json이 없어 이번 probe로 Teacher 전체 source 계약을 인증했다고 주장하지 않는다. 기존 Teacher manifest/contract 검증과 구분한다.
- 준비/Teacher/Student source에서 raw manifest 재사용 시 모든 raw split 파일의 SHA를 읽는 경로가 존재한다. 따라서 test 바이트의 절대 미접근 선언은 부정확하다. 원문 split 생성과 기계적 무결성 읽기는 test 모델 채점·선택 사용과 구분한다.
- Teacher/round1 고정 CLI source의 prepare 단계 packing loop는 train/validation만 처리하며, test packing은 별도 evaluate 경로에 있다. round1 tuning도 train/validation만 packing한다. round2/seed43/seed44의 같은 경로 검토와 source 해시 일치는 이전 증거에 보존돼 있다.
- 검사한 결과 경로에 test-results.json은 없었다. 이는 역사적 evaluate 미실행의 증명이 아니다. 명령 기록 누락·별도 경로 실행 가능성은 이 자료만으로 배제할 수 없다.

## 판정 및 적용 범위
현재 검토한 증거에서는 공통 test를 모델로 채점하거나 선택에 사용한 적극적 증거가 확인되지 않았다. 그러나 **검증된 모든 역사에 걸친 blind test**라고 단정하지 않는다. 공통 raw test의 생성/무결성 접근 사실을 공개하고, 역사적 채점·사람 열람·선택 사용의 완전한 추적은 불확실로 남긴다. pilot의 이미 열린 test와 common cohort를 혼동하지 않는다.

이 판정 때문에 test 문서를 새로 열거나 평가를 반복하지 않았다. 빈 exposure events 배열은 '모든 바이트 비접근'을 뜻하지 않으며, 기존 exposure primitive는 실제 비접근 증거를 인증하지 않는다. 최종 frozen protocol에는 본 판정 문서의 hash와 증거 범위를 별도로 결합해야 한다. 원시 무결성 읽기를 채점 노출로 오인하지도, 누락 기록을 비접근 증명으로 사용하지도 않는다.

## 근거
- `results/research/campaign/common-corpus-v1/final-exposure-history-20261026T1742.json`: 원격 metadata/source 해시 및 line evidence 전체 반환, KD_EXPOSURE_HISTORY_COMPLETE 확인.
- `results/research/campaign/common-corpus-v1/final-exposure-paths-20261026T1724.json`: round2/seed43/seed44. 저장된 stdout의 절단 표시가 있으므로 그 파일만으로 완전한 원문 반환을 주장하지 않는다.
- 기존 original bundle/Teacher full contract 및 cohort binding 검증은 각 원본 증거에 보존한다.

## 다음 유한 작업
1. 추가적인 완전 비접근 증명 탐색을 종료한다. 역사적 launcher의 단계 명시 기록은 가능한 범위에서 원본과 결합하되, 부재로 절대 명제를 만들지 않는다.
2. 평가 결과의 중복 실행 방지/중단 재사용을 vertical TDD로 구현하고 완전한 frozen protocol에 cohort·모델명단·수치정책·본 판정을 결합한다.
3. 독립 리뷰, 새 immutable 배포, 원격 회귀 이후에만 단일 GPU 최종평가를 실행한다. 현재 평가 실행·test 채점은 미완료다.
