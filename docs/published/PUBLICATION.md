# 공개 연구 자료의 범위와 검증

## 공개 구분

- **완료 보고:** [일반 한국어 3계열 KD 연구](generic-kd-v1/README.md).
- **완료 결과물:** [재계산 가능한 집계 JSON·CSV·검증](../../results/published/generic-kd-v1/README.md).
- **공동 재실험 안내:** [4-way 개발자 참고](../FOUR_WAY_REPRODUCTION_GUIDE.md).
- **진행 중 연구:** [한국어 QA 중간보고](taskqa-status-20261105.md). Teacher 적응/A·B·C의 완료 효과는 아직 없다.
- **자료 metadata:** [QA 소스·revision·내부 분할·권리](taskqa-dataset-metadata.json).

위 연구는 과거 소형 모델 결과와 별도다. Qwen/Gemma/Liquid의 raw PPL로 한국어 품질 순위를 만들지 않으며, KD의 소폭 개선을 통계적·실용적·의료적 효과로 확대하지 않는다.

## 포함하지 않는 것

모델 가중치·optimizer payload·HF cache, raw/train/dev/test/sealed 코퍼스와 gold·생성 trace, 접근 토큰/SSH key/환경 비밀, 클라우드 지원 개인정보·운영 로그·중앙 writer state는 공개하지 않는다. 기존 로컬 원본은 그대로 보존한다. Git에 추가한 source helpers와 결과 집계의 공개가 실제 서버·데이터·가중치 접근 권한을 부여하지 않는다.

공통 원문 cohort를 쓰는 재실험은 승인된 원본/manifest에 접근해야 한다. 공식 HF 링크는 원천자료 경로이지 현재 분할의 완전한 재구성 보장이 아니다. 이미 공개된 최종 test는 재현 확인과 신규 튜닝의 독립 test를 구분해서 사용한다. KLUE의 뉴스 원자료는 일반 뉴스/상업 서비스 재사용 권리를 별도 확인한다.

## 코드·환경의 의미

`src/research_*.py`, 수치정책/체크포인트/완료기록 모듈과 학습·튜닝·최종평가 helper를 공개한다. **현재 helper source가 과거 각 실행의 VM immutable capsule 전체와 byte-for-byte 같다는 주장은 아니다.** 역사적 source·선택·checkpoint의 연결은 완료 보고의 근거 범위를 따른다. 연구 결과를 다시 해석하기 위해 원본을 변경하거나 현재 원격 학습을 재시작하지 않았다.

기존 `pyproject.toml`은 4-way/Transformers4 환경을 보존한다. 새 helper에는 별도 `requirements/research-2026.txt` 환경을 쓴다. 핵심 패키지 pin은 이식 가능한 시작점이며 원래 CUDA wheel, GPU, OS, 전체 transitive lock과 같음을 보증하지 않는다. `configs/research_*.yaml`은 20-step GPU smoke용이지 최종 실험 재현 plan이 아니다.

CPU 회귀에는 작성된 작은 모델/fixture를 사용한다. 통과가 실제 GPU 효과·Blob 전체 내구성·실기기 품질을 새로 인증하지는 않는다. 공개 준비 과정에서 기존 테스트의 `fstat` 오류 주입을 해당 결과 파일 descriptor로 제한하여 pytest의 출력 캡처까지 망가뜨리던 fixture 문제를 수정했다. 생산 학습·평가 수치 코드는 이 수정으로 바꾸지 않았다.

공개 회귀 테스트에 필요한 역사적 계약은 `tests/fixtures/generic-final-metadata/`에 메타데이터 projection 또는 byte-identical metadata 사본으로 분리했다. 원래 archive 전체가 아니며, 원문·가중치·실행 권한을 담지 않는다. 각 원본 파일과 보존 node의 SHA 및 정확한 projection 범위는 해당 폴더의 `PROVENANCE.json`과 README에 있다. 기존 revision/hash/선택 검사는 유지하고 경로만 공개 fixture로 바꿨다.

## 공개 체크아웃 검증

비공개 `results/research/`가 없는 별도 index export에서 연구 회귀 **313 passed /155 subtests passed**, 공개 결과 검증기 **24 results /9 pairs /21 selected metadata**, 검증기 자체 **4 tests OK**를 확인했다. CLI 세 개의 `--help`도 모델·코퍼스 취득 없이 성공했다. 이는 해당 범위의 CPU/집계 검증이며 전체 legacy 환경 테스트나 실제 GPU 재학습·새 QA 채점을 뜻하지 않는다.

격리 연구 환경을 준비한 뒤 저장소 루트에서:

```bash
.venv-research/bin/python -m pytest \
  tests/test_research*.py tests/test_checkpoint_blob.py \
  tests/test_training_resume.py tests/test_stage_records.py \
  tests/test_numerical_policy.py tests/test_tuning_service.py \
  tests/test_final*.py -q
.venv-research/bin/python -B results/published/generic-kd-v1/verify_results.py
.venv-research/bin/python -B results/published/generic-kd-v1/test_verify_results.py
```

공개 결과와 fixture의 hash-bound bytes는 `.gitattributes`로 checkout 줄바꿈 변환을 막는다. 원본 paired CSV의 CRLF는 의도적으로 보존했다.

## 출처 보존

[source-provenance.json](source-provenance.json)은 기초 최종보고·QA 보고초안·시간이 고정된 인계 문서 등의 원래 bytes/SHA를 기록한다. 외부 작업공간의 prefix는 공개용 논리 이름이며 파일이 이 저장소에 그대로 포함됐다는 뜻이 아니다. QA 중간보고는 완료 집계 수치와 저장 관측의 설명을 옮긴 것으로, 과거 source/raw/trace의 신규 전체 검증은 아니다.

기초 결과 패키지의 자체 README/manifest는 그 패키지의 원본 집계 파일·공개 projection·재계산 및 포함/제외 범위를 명시한다. 원문 문서의 역사적 절대경로나 private Blob 경로는 공개 다운로드 주소가 아니다. 재현 가능한 공개 경로는 결과 패키지의 portable index를 따른다.
