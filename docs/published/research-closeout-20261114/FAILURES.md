# 미개선·실패·미완료를 구별한 원인 분석

**분석 원칙:** 측정한 성능 악화는 결과이고, 그 원인은 별도 검증 대상이다. 인프라 장애로 실험을 끝내지 못한 경우에는 “KD가 효과 없음”으로 쓰지 않는다. 아래 진단은 당시 기록의 회고이며 현재 클라우드를 새로 실행해 확인한 결과가 아니다.

## 판정 사전

| 구분 | 의미 | 이 연구에서의 예 |
|---|---|---|
| 측정상 개선 | 같은 cohort·평가기·비교조건에서 지표가 낮아짐 | 정식 Qwen 세 seed의 CE→KD test PPL 감소 |
| 측정상 악화/혼재 | 유효하게 측정했으나 대조군을 이기지 못하거나 seed 간 방향이 다름 | Gemma 두 seed 악화, Liquid 한 seed 악화 |
| 품질 gate 실패 | 실행은 됐지만 목표 답변 품질 기준을 충족하지 않음 | 비의료 Teacher dev 검토 11건 중 수정 필요 8건 |
| 실행 실패 | 로더·dtype·timeout 등으로 해당 실행/평가가 유효하게 끝나지 않음 | processor ImportError, 일부 evaluator dtype 오류 |
| 환경·데이터 준비 차단 | 실행에 필요한 자원/자료/호환성 근거가 확보되지 않음 | Spot 회수, 자료 접근 오류, 후속 서비스 후보 검증 미완료 |
| 미실시/미완료 | 비교나 평가가 완성되지 않음 | 임상 QA 및 저사양 기기 검증 |
| 의도적 중지 | 사용자 요청에 따른 종료 | 자동 실행·보고 중지. 성능/수렴 판정과 별개 |

## 1. KD가 대조군을 항상 이기지 못했다

기본 3계열 실험에서는 Qwen은 세 Student seed에서 −0.26%∼−0.54% 정도 개선했으나 Gemma·Liquid는 개선과 악화가 혼재했다. 이는 유한 예산을 마친 **유효한 부정/혼재 결과**다. Spot 오류가 있으니 이 결과를 무효로 하거나, 반대로 상한 step에 도달했으니 충분히 수렴했다고 해석하지 않는다. 정확한 per-seed 값·계약·선택은 [기존 최종보고](../generic-kd-v1/REPORT.md)와 [공개 집계](../../../results/published/generic-kd-v1/README.md)에 보존돼 있다.

**가능한 설명과 확인 범위:** Liquid의 같은 test에서 고정 Teacher PPL `19.003929`는 최종 CE/KD Student보다 높았다. Teacher가 그 지표에서 더 강하지 않았다는 것은 확인됐지만, 그것이 혼재 결과의 단일 인과 원인이라고 검증한 것은 아니다. Gemma의 높은 초기 평문 PPL도 특정 loader 결함으로 확정되지 않았다. Teacher 적응, task 적합성, KD 가중치·온도·방향, 작은 추가 효과와 seed 변동은 후속 진단 가설이다. 대규모 추가 탐색이나 인과 ablation을 실시한 것처럼 쓰지 않는다.

실버케어 보조 대화 matched12에서도 B(Teacher 응답 CE)보다 C(CE+KL)는 좋아졌지만 A(정답 CE)는 이기지 못했다. 한 seed·12 update·내부 dev 결과이고, 임상 효능 실패를 측정한 결과가 아니다. [전체 수치](SILVERCARE.md#3-12행-matched-abc--kl-효과와-정답-ce-비교를-분리).

## 2. 손실 감소가 실제 응답 품질을 보장하지 않았다

41행 비의료 Teacher 적응의 dev PPL은 낮아졌지만 실제 내부 dev 검토에는 감정 불일치, 사건 재진술 위주, 근거 없는 원인·동기 부여와 반복 등이 남았다. 따라서 “학습이 안 됐다”보다 **학습 지표는 개선했으나 목표 응답 품질 gate는 실패했다**가 정확하다. 사람이 검수한 임상 gold나 의료 전문가 평가가 아니므로 검토 비율을 의료 정확도로 발표하지 않는다.

후속 정책 전후 dev11의 24-call LLM 검토도 전체 품질 향상을 입증하지 못했다. 한편 **별도 train41의 43호출 검토(41train+2neutral) 뒤 12응답은 비의료 per-case 학습에 한해 수용**되어 실제 matched12 실행으로 이어졌다. 초기 dev 불수용과 뒤의 제한 수용은 서로 다른 단계다. Teacher 전체 우위·임상 승인은 여전히 미검증이다. 상세 집계·후속 PEFT3/Azure 후보 단일 추론의 작은 성공은 [서비스 결과](SILVERCARE.md)에 함께 보존했다.

후속 matched12 답변 33건 중 32건이 EOS 없이 64토큰 상한에 도달했고, 완전한 의미 비교는 미완료다. 상한을 늘려서 긍정 결과를 찾은 것처럼 덮지 않고 실제 종료 metadata와 미완료 상태를 보존한다.

## 3. Azure Spot: 실제 자원 회수와 소프트웨어 실패가 겹쳤다

`environment-results.json`의 `spot-and-loader`는 2026-11-09의 제한된 진단 문서에서 정확한 줄을 추출했다.

- 해당 24시간 Activity Log 관측에는 Spot 회수 8회 / 시작 성공 8회 / 시작 실패 7회가 기록됐다. 모든 회수의 세부 capacity 원인을 확정한 것은 아니다. 이 수는 전체 연구 기간의 총계가 아니다.
- `maxPrice=-1`에서도 capacity 회수는 가능했다. GPU가 Spot-only인 조건은 유지됐고 Regular GPU를 실제 사용한 대안으로 쓰지 않는다.
- 별도로 PROCESSOR ImportError, 불필요한 visual/MTP 로더 분기, 300초 alarm/systemd timeout 이후 SIGKILL이 있었다. **모든 중단이 Spot 때문인 것은 아니다.**
- CPU에서 해둘 준비·소스 진단과 GPU 복원이 직렬화되어, Running 상태가 실제 추론/update로 이어지지 않는 구간이 있었다.

이 진단은 당시 Azure 서비스 경로다. 이후 다른 작업공간에서 실제 학습/생성이 완료된 사실을 지우고 “연구 전체의 실추론이 0이었다”고 일반화하면 안 된다.

## 4. 후속 학교 서비스 후보: 제공자 금지와 애플리케이션 설계 blocker를 구별

기존 비공개 단일 요청 CE41 E2E의 성공 기록과 후속 상시 서비스 후보는 다른 경로다. 후속 후보는 systemd/cgroup, 모든 descendant 종료, cache publisher custody, 동시 RAM/VRAM peak, 실제 모델·tokenizer constructor identity 등의 검증이 닫히지 않았다. 이들은 해당 후보의 구현·안전성 요건 또는 미확인 사항으로, 해당 12-file 감사에서는 **제공자의 학습/서비스 금지 근거가 성립하지 않았다**.

- systemd가 없다는 관측 ≠ 모든 실행이 정책상 금지됐다는 의미.
- 기존 단일 요청 E2E 성공 ≠ 새로운 지속 서비스·전환·rollback의 성공.
- 디스크 50GiB 예산 ≠ RAM 50GiB 상한.
- reduced-shape MPS fixture PASS ≠ 실제 4B pretrained 가중치 복원 및 서비스 성공.

해당 경계·메모리·구성요소 검증을 모두 해결했다고 쓰지 않으며, 관리자에게 요청할 초안과 실제 발송/회신도 구별한다. `environment-results.json`의 `application-not-provider-ban`에 원문 감사의 한정된 근거를 보존했다.

## 5. 데이터 접근·연결·수용이 학습 확대를 제한했다

2026-11-09의 추가 자료 접근 점검에서 dataset `71852`, `576`, `470`은 HTTP502였고 정상 payload 취득은 확인되지 않았다. 과거 성공 자료의 control도 이번에는 HTTP404였다. 키 무효·만료·승인 취소 중 어느 것이 원인인지 이 응답만으로 확정하지 않는다. 이는 **그 시점의 취득 실패**이지 자료의 영구 이용 불가 판정이 아니다. status/selector/byte-count만 공개하고 자격증명은 제외한다.

자료 카탈로그 조사, 파일 취득·무결성, QA 연결·의미 수용, 임상 수용, split/tokenization, Teacher 검토, 실제 학습은 각각 다른 단계다. 원본 파일이 있어도 임상 QA 전체가 학습 준비됐다는 뜻은 아니다. 의미/연결이 미결인 자료를 임의 pairing하거나 다운로드 성공을 모델효능으로 세지 않는다.

## 6. 복구 시험에서 확인한 것과 확인하지 못한 것

보존된 CPU 합성 SIGKILL 경계 시험 두 건은 payload 쓰는 중 및 COMPLETE 이후 latest 게시 전에 kill=-9를 주고, 마지막 게시 step0에서 재개해 final step4에 도달했다. 모델/optimizer/RNG/cursor/history 등은 uninterrupted control과 일치했다. 원래 source hash도 불변이다. `environment-results.json`의 `cpu-sigkill-checkpoint`에 결과를 보존했다.

이 성공은 **CPU 합성 checkpoint 게시/재개 경계**에 한정된다. 실제 Azure Spot eviction, GPU 실제 모델의 수치 동일성, 전원손실, 모든 Blob payload 내구성, 실제 서비스 복구를 이번 결과로 대신 인증하지 않는다. 이미 완료된 별도 복구 증거가 있다면 그 원래 scope를 따른다.

## 결론을 과장하지 않는 문장

- 허용: “고정 Teacher와 세 Student seed 조건에서 Qwen의 test PPL은 CE 대비 소폭 감소했다.”
- 허용: “Gemma·Liquid는 seed별 효과 방향이 혼재하여 보편적 KD 우위를 확인하지 못했다.”
- 허용: “보조 대화의 KL 추가는 응답 CE보다 dev PPL을 개선했으나 정답 CE는 넘지 못했다.”
- 허용: “일부 후속 비교·서비스 검증은 환경/구현·자료/품질 gate 미해결 및 사용자 중지로 완료되지 않았다.”
- 금지: “환경 때문에 실패했으므로 실행했으면 KD가 더 좋았을 것이다.” / “PPL이 좋아졌으니 의료·모바일 서비스가 검증됐다.” / “미실시 비교의 개선율은 0%다.”
