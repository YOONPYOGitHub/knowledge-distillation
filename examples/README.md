# CIFAR-10 Knowledge Distillation 실습 예제

> CNN 기반 CIFAR-10 이미지 분류에서 **Logit-based Knowledge Distillation**을 실습하는 교육용 노트북.  
> 본 프로젝트(LLM 지식 증류)의 핵심 개념을 작은 규모로 먼저 체험해 볼 수 있습니다.

---

## 노트북 소개

### `cifar10_knowledge_distillation.ipynb`

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

#### 노트북 흐름

| 단계 | 셀 | 내용 |
|------|-----|------|
| **준비** | 1~3 | Import, 디바이스(GPU/CPU) 설정 |
| **데이터** | 4~6 | CIFAR-10 다운로드, 전처리, DataLoader 생성 |
| **모델 정의** | 7~8 | Teacher(`DeepNN`) / Student(`LightNN`) 클래스 정의 |
| **유틸 함수** | 9~10 | 일반 학습(`train`) / 평가(`test`) 함수 정의 |
| **Teacher 학습** | 11~12 | DeepNN 10 에폭 학습 → 정확도 확인 |
| **Student 준비** | 13~14 | 동일 seed로 Student 2개 생성 (공정 비교용), 파라미터 수 비교 |
| **Student 단독 학습** | 15~16 | Hard Label만으로 학습 (baseline) |
| **개념 설명** | 17~18 | Soft Target / Dark Knowledge / KD 필요성 설명 |
| **Knowledge Distillation** | 19~20 | KD 학습 함수 정의 + 실행 → **3-Way 결과 비교** |

#### 핵심 Loss 함수

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

## 본 프로젝트와의 관계

| 항목 | 이 예제 (CIFAR-10) | 본 프로젝트 (LLM) |
|------|---------------------|---------------------|
| 도메인 | 이미지 분류 | 언어 모델링 |
| Teacher | DeepNN (커스텀 CNN) | GPT-2 Large (774M) |
| Student | LightNN (커스텀 CNN) | GPT-2 Small (124M) |
| 데이터셋 | CIFAR-10 (6만장) | WikiText-2 |
| KD Loss | KL Div + CE | KL Div + CE (동일 원리) |
| 비교 방식 | 3-Way | 4-Way (FT baseline 추가) |

> 이 예제에서 KD의 원리를 이해한 뒤, 본 프로젝트의 LLM 증류로 확장하는 것을 권장합니다.

---

## 실행 방법

### 사전 요구사항

- Python 3.13+ (uv 환경 설정 완료)
- GPU 선택사항 (CPU에서도 동작, 학습 시간 더 소요)

### 1. 의존성 설치

```bash
cd examples
uv sync
```

### 2. 노트북 실행

1. VS Code에서 `cifar10_knowledge_distillation.ipynb` 열기
2. 우측 상단에서 커널을 `.venv` Python 인터프리터로 선택
3. **Cell 1부터 순서대로 실행** (Run All 또는 셀 하나씩)

> CIFAR-10 데이터셋(~170MB)은 첫 실행 시 `./data/` 폴더에 자동 다운로드됩니다.

### 3. 실행 시간 참고

| 환경 | 예상 소요 시간 |
|------|---------------|
| GPU (CUDA) | ~5분 |
| Apple Silicon (MPS) | ~10분 |
| CPU | ~30분+ |

> CPU만 사용할 경우 각 학습 셀의 `epochs` 값을 줄여서 먼저 실습해 보세요.

---

## 참고

- 원본 논문: Hinton et al. (2015) — *"Distilling the Knowledge in a Neural Network"*
- 상위 프로젝트: [README.md](../README.md)
