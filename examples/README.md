# Knowledge Distillation 프로젝트

Knowledge Distillation(지식 증류)의 원리를 학습하고, LLM → SLM 지식 전이를 실습하는 프로젝트.

---

## 노트북 소개

### `[필로소피_AI_교육]_Knowledge_Distillation_지식_증류.ipynb`

CIFAR-10 이미지 분류 태스크에서 **Logit-based Knowledge Distillation**을 실습하는 교육용 노트북.

#### 핵심 개념

큰 모델(Teacher)의 **Soft Target(확률 분포)**을 작은 모델(Student)이 모방하도록 학습하여, 작은 모델의 성능을 끌어올린다.

```
Teacher (DeepNN)  ──→  Soft Labels (확률 분포)  ──→  Student (LightNN)
                                                      + Hard Labels (정답)
```

#### 모델 구성

| | Teacher (`DeepNN`) | Student (`LightNN`) |
|---|---|---|
| 종류 | PyTorch `nn.Module` 커스텀 CNN | PyTorch `nn.Module` 커스텀 CNN |
| Conv 레이어 | 4층 (128→64→64→32 채널) | 2층 (16→16 채널) |
| FC 레이어 | 2048→512→10 | 1024→256→10 |
| 역할 | 높은 정확도의 큰 모델 | 가볍지만 KD로 성능 향상을 노리는 작은 모델 |

> 두 모델 모두 라이브러리가 아니라 **노트북 안에서 직접 정의한 신경망 클래스**입니다.

#### 셀 실행 순서 및 내용

| 셀 | 내용 | 설명 |
|----|------|------|
| 1~3 | Import & Device 설정 | PyTorch, torchvision 로드. GPU/CPU 자동 감지 |
| 5~6 | 데이터 전처리 & 다운로드 | CIFAR-10 (32x32 이미지 6만장, 10클래스) 자동 다운로드 |
| 7~8 | DataLoader | batch_size=128로 데이터 로더 생성 |
| 9~10 | Teacher 모델 정의 | `DeepNN` 클래스 (Conv 4층 + FC 2층) |
| 11~12 | Student 모델 정의 | `LightNN` 클래스 (Conv 2층 + FC 2층) |
| 13~14 | 일반 학습 함수 | CrossEntropyLoss + Adam (Hard Label 학습) |
| 15~16 | 평가 함수 | 테스트셋 정확도 측정 |
| 17~18 | Teacher 학습 | DeepNN을 10 에폭 학습 |
| 19~20 | Student 2개 생성 | 같은 seed로 초기화 (공정 비교용) |
| 21~22 | 파라미터 수 비교 | Teacher vs Student 크기 차이 확인 |
| 23~24 | Student 단독 학습 | Hard Label만으로 학습 (baseline) |
| 25~26 | 성능 비교 & 개념 설명 | Soft Target / Dark Knowledge 설명 |
| 27~28 | **Knowledge Distillation** | Temperature=2, KL Divergence로 Teacher→Student 지식 전이 |

#### 핵심 Loss 함수 (Cell 28)

$$L_{total} = 0.25 \times L_{KL}(Teacher, Student) \times T^2 + 0.75 \times L_{CE}(Student, Label)$$

- **Temperature (T=2)**: logits를 T로 나눠 확률 분포를 부드럽게 만듦 → Dark Knowledge 전달
- **KL Divergence**: Teacher와 Student의 확률 분포 차이를 최소화
- **CrossEntropy**: 실제 정답 레이블도 함께 학습

#### 기대 결과

```
Teacher 정확도     > KD Student 정확도     > 단독 Student 정확도
(DeepNN ~75%)       (LightNN+KD ~71%)       (LightNN ~70%)
```

---

## 실행 방법

### 사전 요구사항

- Python 3.13+ (uv 환경 설정 완료)
- GPU 선택사항 (CPU에서도 동작, 학습 시간 더 소요)

### 1. 의존성 설치

```bash
# .venv 생성 + 패키지 설치
uv sync
```

### 2. 노트북 실행

1. VS Code에서 `[필로소피_AI_교육]_Knowledge_Distillation_지식_증류.ipynb` 열기
2. 우측 상단에서 커널을 `.venv` Python 인터프리터로 선택
3. **Cell 1부터 순서대로 실행** (Run All 또는 셀 하나씩)

> CIFAR-10 데이터셋(~170MB)은 첫 실행 시 `./data/` 폴더에 자동 다운로드됩니다.

### 3. 실행 시간 참고

- GPU(CUDA) 환경이 CPU보다 학습 속도가 훨씬 빠릅니다.
- CPU만 사용할 경우 학습 시간이 길어질 수 있으니, 필요하면 각 학습 셀의 `epochs` 값을 줄여서 먼저 실습한 뒤 점진적으로 늘려보세요.

---

## 참고

- 원본 논문: Hinton et al. (2015) - "Distilling the Knowledge in a Neural Network"
