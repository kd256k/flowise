# Flowise — LSTM 수요 예측 모델 개발 기록

## 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 프로젝트 | Flowise — 정수장 에너지 최적화를 위한 수요 예측 AI |
| 대상 | J배수지 (reservoir/10.csv) |
| 목표 | 1분 단위 유량의 15분 선행 예측 (×4 rolling = 1시간) |
| 데이터 | 유량 943,434건 + 기상 1,885,020건 (2023-01 ~ 2024-10) |

---

## 개발 타임라인

```
Phase 1  모델 탐색 (12개 아키텍처 실험)
  ├─ segment_id 버그 발견 → clean 파이프라인 재실험
  ├─ 12개 모델 확정 순위 → Best: Seq2SeqAttn (MAPE 2.79%)
  └─ 실험 보고서: 12model_benchmark_report.md

Phase 2  Best Model 개선 (v2 Revision)
  ├─ 8가지 개선 적용 → 성능 2배 악화 (2.79% → 5.82%)
  ├─ v2 분석 보고서 작성 (4개 그래프 포함)
  └─ Ablation Study 설계 (A/B/C)

Phase 3  원인 규명 (Ablation Study)
  ├─ Ablation A: Rainfall 원복 → 원인 아님 확인 (5.92%)
  ├─ Ablation B: Dropout 제거 → 원인 확정, v1 복원 (2.89%)
  └─ 최종 모델 v3 확정 (MAPE 2.89±0.31%)
```

---

## Phase 1: 모델 탐색

### 1.1 실험 설계

12개 LSTM 기반 아키텍처를 동일 조건에서 비교하여 최적 구조를 탐색.

**실험 목적 8가지:**

1. Encoder-Decoder (Seq2Seq) 구조의 유효성 검증
2. Bahdanau Attention의 Seq2Seq 적용 효과 확인
3. Autoregressive (prev_pred) vs Step Embedding 비교
4. 60일 데이터 확장 효과 확인 (45d → 60d)
5. Step 열화 분석 (먼 step의 예측 정확도 감소 정량화)
6. Multi-Head FC 구조 실험 (step별 독립 FC head)
7. 디코딩 방식별 비교: FC 동시 예측 / Residual FC / Cross-Attention / Decoder LSTMCell
8. Loss 함수 영향 비교: Step-weighted MSE / MSELoss / HuberLoss

### 1.2 공통 설정

| 항목 | 값 |
|------|-----|
| Input | 72분 (1.2시간), 10 features |
| Output | 15분 (× 4 rolling = 1시간) |
| Features | value, temperature, rainfall, humidity + 6 cyclical temporal |
| Data Split | Train 70% / Val 15% / Test 15% (block별, gap=87) |
| 정규화 | Train 기준 MinMaxScaler (sin/cos 제외) |
| Early Stopping | patience=5 |
| Optimizer | Adam (lr=0.001, weight_decay=1e-5) |
| Gradient Clipping | max_norm=1.0 |

### 1.3 데이터 파이프라인

**전처리:**

```
Flow:    943,434건 → IQR 이상치 제거(962건) → Linear Interpolation → Savitzky-Golay
Weather: 1,885,020건 → 결측 보간 → 장기 결측 제거 → 894,039건
Merge:   Inner join → 872,723건 → Temporal features → 323,308건 (60d×4계절)
```

**계절별 Block Sampling (60일 × 4):**

| 계절 | 기간 |
|------|------|
| Winter | 2023-12-01 ~ 2024-01-29 |
| Spring | 2024-03-15 ~ 2024-05-13 |
| Summer | 2023-07-01 ~ 2023-08-29 |
| Fall | 2023-09-15 ~ 2023-11-13 |

**segment_id 버그 수정:**

| 항목 | Buggy (수정 전) | Clean (segment-aware) |
|------|:---:|:---:|
| 60d 총 윈도우 | 322,268 | **239,726** |
| 오염 윈도우 비율 | 25.56% | 0% |
| 세그먼트 경계 처리 | 무시 | segment별 독립 생성 |

시간 불연속 경계를 넘는 sliding window가 전체의 25.56%를 오염시키고 있었음.
clean 파이프라인으로 전 모델 재실험하여 결과 확정.

### 1.4 전체 결과 순위 (Clean)

| 순위 | 모델 | 데이터 | MAPE | R² | Bias | Epoch | 파라미터 |
|:----:|------|:------:|-----:|----:|-----:|------:|--------:|
| **1** | **Seq2SeqAttn 60d** | 60d | **2.79%** | **0.9849** | +0.86 | 53 | 377,585 |
| 2 | Seq2Seq 60d | 60d | 3.39% | 0.9831 | -0.66 | 23 | 361,201 |
| 3 | Residual 45d | 45d | 3.70% | 0.9799 | -1.21 | 37 | 205,967 |
| 4 | Residual 60d | 60d | 3.98% | 0.9797 | +0.07 | 19 | 205,967 |
| 5 | StepAttn_mid 60d | 60d | 4.01% | 0.9768 | +3.06 | 16 | 274,064 |
| ※6 | Baseline | 비표준 | 4.33% | 0.9739 | -2.75 | 19 | 205,711 |
| 7 | StepAttnModel 45d | 45d | 4.49% | 0.9708 | +3.78 | 26 | 272,129 |
| 8 | StepAttn_mid 45d | 45d | 4.56% | 0.9716 | +4.40 | 16 | 274,064 |
| ※9 | StepAttn_high | 60d† | 7.26% | 0.9126 | +7.61 | 16 | 272,129 |
| 10 | StepEmbed 60d | 60d | 9.20% | 0.9040 | +14.18 | 6 | 208,569 |
| 11 | MultiHead 60d | 60d | 10.77% | 0.8858 | +16.28 | 9 | 266,447 |
| 12 | Autoreg 45d | 45d | 14.30% | 0.8412 | -10.24 | 9 | 369,409 |

> ※6 Baseline: 비표준 파이프라인(weather 22파일, log1p rainfall, gap 없음)
> ※9 StepAttn_high: 비표준(Fall 범위 상이 + HuberLoss)

### 1.5 핵심 발견

**아키텍처 계층:**

```
Decoder + Attention  (MAPE 2.79%)
  > Decoder only     (3.39%)
    > Residual FC    (3.70~3.98%)
      > Cross-Attn   (4.01~4.56%)
        > FC only    (4.33%)
          > 실패 그룹 (9.20~14.30%)
```

**발견 1 — Seq2Seq Encoder-Decoder 구조 유효:**
Decoder LSTMCell의 순차 디코딩이 FC(128→15) 동시 예측보다 우수. 상위 2개 모두 Decoder 구조.

**발견 2 — Bahdanau Attention 유효 (조건부):**
동적 Query(decoder_hidden_t) 사용 시에만 유효. 고정 Query(step_queries)를 사용하는 Cross-Attention은 weight가 균등(≈1/72)으로 수렴하여 무효.

**발견 3 — step_embedding > prev_pred:**
Autoregressive의 prev_pred는 오류 누적(+1.804%/step)을 유발하여 Step 15 MAPE 27.43%로 실패. step_embedding은 오류 차단으로 +0.328%/step 유지.

**발견 4 — 비선형 Decoder 실패:**
MultiHead FC(`[Linear(128→32→ReLU→1)] × 15`)와 StepEmbed(`Linear(136→32→ReLU→1)` shared) 모두 실패. ReLU를 포함한 비선형 decoder는 이 데이터에서 underfitting.

**발견 5 — Step-weighted MSE 효과적:**
step 1~15에 가중치 1.0~2.0 선형 배정. 상위 5개 모델 중 4개가 채택.

**발견 6 — Step 열화 패턴:**

| 패턴 | 모델 | 특징 |
|------|------|------|
| 정상 | Seq2SeqAttn, Seq2Seq, Residual | Step 증가에 따라 단조 증가 |
| 역전 | StepEmbed, MultiHead | Step 순서 무관한 불규칙 MAPE → step 정체성 미학습 |
| 폭발 | Autoreg | +1.804%/step, prev_pred 피드백 루프의 구조적 한계 |

---

## Phase 2: v2 Revision

### 2.1 개선 시도

v1(Best Model)에 8가지 개선 사항을 동시 적용:

| # | 변경 | 내용 | 의도 |
|:-:|------|------|------|
| 1 | Multi-run | N=3, seeds=[42, 123, 7] | 결과 신뢰성 확보 |
| 2 | Bias 진단 | Scaler 범위, 계절·Step·값 구간별 분해 | 체계적 편향 파악 |
| 3 | patience 확대 | 5 → 10 | 더 깊은 수렴 기회 |
| 4 | Decoder dropout | 0.2 추가, Drop→LN→FC 순서 | 일반화 성능 향상 |
| 5 | Rainfall 차분 | 일 누적 → 분당 변화량 | feature 품질 향상 |
| 6 | Dead code 제거 | `_bahdanau_attention()` 삭제 | 코드 정리 |
| 7 | GPU 메모리 | CPU 텐서 + pin_memory | OOM 방지 |
| 8 | 평가 지표 | SMAPE, macro-avg MAPE 추가 | 평가 보강 |

### 2.2 v2 결과 — 성능 2배 악화

| 지표 | v1 (단일) | v2 (Mean±Std) |
|------|:---------:|:-------------:|
| MAPE | **2.79%** | 5.82 ± 0.33% |
| R² | **0.9849** | 0.9714 ± 0.002 |
| Bias | +0.86 | +0.78 ± 0.62 |
| Epoch | 53 | 31 (평균) |

### 2.3 v2 분석 결과

**v2 분석 보고서** (seq2seq_attn_v2_analysis.md)에서 도출한 주요 발견:

1. **Step 열화 기울기 27% 개선** — +0.328 → +0.241/step. 다만 Step 1 MAPE가 1.14%→4.45%로 3.9배 악화되어 전체 MAPE는 오히려 악화.

2. **Bias의 "regression to mean" 패턴** — 저유량 +7.05, 고유량 -7.02로 대칭적 수축. MSE Loss의 본질적 특성.

3. **Attention 차별화 완전 실패** — v1에서도 동일. 15 step 모두 Entropy 4.191로 동일한 recency-biased 분포.

4. **seed 의존성** — Best(5.35%) vs Worst(6.10%)로 0.75%p 차이.

### 2.4 원인 가설 수립

8가지 동시 변경 중 성능 하락 원인 후보:

| 가설 | 변경 사항 | 검증 방법 |
|------|-----------|-----------|
| H1 | Rainfall 차분 변환 | Ablation A (rainfall 원복) |
| H2 | Decoder dropout + LN 순서 | Ablation B (dropout 제거) |
| H3 | v1의 2.79%가 lucky run | Ablation C (v1 구조 multi-run) |

---

## Phase 3: Ablation Study

### 3.1 실험 설계

| 실험 | Rainfall | Decoder Dropout | LN 순서 | 목적 |
|:----:|:--------:|:---------------:|:-------:|------|
| v2 (baseline) | 차분 | 0.2 | Drop→LN→FC | 비교 기준 |
| **A** | **누적 (원복)** | 0.2 | Drop→LN→FC | H1 검증 |
| **B** | 차분 | **없음 (원복)** | **LN→Drop→FC** | H2 검증 |
| C | 누적 (원복) | 없음 (원복) | LN→Drop→FC | H3 검증 (불필요로 미실행) |

### 3.2 Ablation A 결과 — Rainfall ≠ 원인

| | v2 | A (rainfall 원복) | 판정 |
|:---:|:---:|:---:|:---:|
| Mean MAPE | 5.82 ± 0.33% | 5.92 ± 0.07% | 차이 없음 |
| Mean R² | 0.9714 | 0.9705 | 동등 |
| Mean Bias | +0.78 ± 0.62 | +0.42 ± 1.20 | 유사 |

**결론:** Rainfall 변환(누적→차분)은 성능 하락의 원인이 아님. **H1 기각.**

### 3.3 Ablation B 결과 — Dropout = 유일한 원인 (확정)

| | v1 | v2 | **B (dropout 제거)** |
|:---:|:---:|:---:|:---:|
| Mean MAPE | 2.79% (단일) | 5.82 ± 0.33% | **2.89 ± 0.31%** |
| Best MAPE | 2.79% | 5.35% | **2.66%** |
| Mean R² | 0.9849 | 0.9714 | **0.9854** |
| Mean Bias | +0.86 | +0.78 | **+0.28** |
| Mean Epoch | 53 | 31 | **58** |

**Run별 상세:**

| Seed | MAPE | R² | Bias | Epoch |
|:----:|-----:|---:|-----:|------:|
| 42 | 2.69% | 0.9863 | +0.38 | 62 |
| 123 | 3.32% | 0.9830 | +0.47 | 39 |
| **7** | **2.66%** | **0.9868** | **-0.02** | **73** |

**결론:** Decoder dropout 제거만으로 v1 수준 완전 복원. **H2 확정.**
v1의 2.79%는 정상 분포(2.66~3.32%) 안의 전형적인 값. **H3도 기각.**
Ablation C는 불필요하여 미실행.

### 3.4 Decoder Dropout이 치명적인 메커니즘

Decoder LSTMCell은 autoregressive하게 15번 순차 실행. Dropout이 h_dec에 적용되면:

1. Step t에서 꺼진 뉴런의 정보가 Step t+1 이후 영구 소실
2. 매 step마다 다른 뉴런이 꺼져 hidden state 연속성 파괴
3. 15-step에 걸쳐 dropout 효과가 누적

추가로 v2의 Drop→LN 순서는 0이 된 뉴런을 포함한 벡터를 LayerNorm으로 정규화하여 분포를 이중 왜곡.

**정량적 증거:**

| 지표 | v2 (dropout on) | B (dropout off) |
|:---:|:---:|:---:|
| Step 1 MAPE | 4.45% | **1.13%** (3.9× 개선) |
| 학습 Epoch | 31 | **58** (1.9× 더 깊은 수렴) |
| Val Loss (best) | 0.001214 | **0.000766** (37% 개선) |

---

## 최종 확정 모델: v3

### 구성

```
v1에서 유지:
  - Decoder: dropout 없음, LN→Dropout→FC 순서

v2에서 유지:
  - Multi-run (N=3) + seed 고정 → 신뢰구간 확보
  - EarlyStopping patience=10 → 더 깊은 수렴
  - pin_memory → GPU 효율
  - SMAPE, Macro MAPE → 평가 보강
  - Rainfall 차분 → 성능에 영향 없으므로 무관

아키텍처:
  Encoder LSTM(10→128, 2-layer, dropout=0.2)
  → Bahdanau Attention (Query=decoder_hidden_t)
  → Decoder LSTMCell × 15 (step_embedding 16-dim)
  Loss: Step-weighted MSE (1.0~2.0)
  파라미터: 377,585개
```

### 확정 성능

| 지표 | Mean ± Std | Best (seed=7) |
|------|:----------:|:-------------:|
| MAPE | 2.89 ± 0.31% | **2.66%** |
| SMAPE | 2.89 ± 0.31% | 2.66% |
| R² | 0.9854 ± 0.0017 | **0.9868** |
| RMSE | 6.83 ± 0.38 | 6.51 |
| MAE | 4.34 ± 0.38 | 4.06 |
| Bias | +0.28 ± 0.21 | **-0.02** |
| Macro MAPE | 2.87 ± 0.29 | 2.63 |

### Step-wise 성능

| Step | MAPE (Mean±Std) |
|-----:|:---------------:|
| 1 | 1.13 ± 0.20% |
| 5 | 1.68 ± 0.17% |
| 10 | 3.44 ± 0.40% |
| 15 | 5.58 ± 0.58% |

### 계절별 성능

| 계절 | MAPE (Mean±Std) |
|------|:---------------:|
| Winter | 2.72 ± 0.18% |
| Spring | 2.85 ± 0.44% |
| Summer | 3.03 ± 0.33% |
| Fall | 2.86 ± 0.30% |

---

## 알려진 한계

| 한계 | 상세 | 영향 |
|------|------|------|
| Attention 차별화 실패 | 15 step 모두 동일한 recency-biased 분포 (Entropy 4.191) | Step 열화 기울기 개선 불가 |
| Regression to mean | 저유량 과대 예측, 고유량 과소 예측 (MSE Loss 본질) | Bias 패턴은 존재하나 전체 Bias +0.28로 실용적 |
| Seed 의존성 | MAPE 2.66~3.32% (0.66%p 범위) | 배포 시 best seed 선택 또는 앙상블 고려 |
| 단일 배수지 | J배수지만 학습/평가 | 다른 배수지 일반화 미검증 |

---

## 핵심 교훈

### 1. Autoregressive Decoder에 dropout을 적용하면 안 된다

Encoder LSTM의 inter-layer dropout(0.2)은 정상 작동.
Decoder LSTMCell의 hidden state에 dropout 적용 시 15-step 순차 실행에서 정보 소실이 누적되어 성능 2배 악화.

### 2. 단일 실행 결과는 신뢰할 수 없다

v1의 2.79%만으로는 lucky run 여부 판단 불가.
v3의 multi-run(2.66~3.32%)으로 정상 분포임을 확인.

### 3. Ablation을 통한 인과관계 확립이 필수

8가지 동시 변경 → 원인 특정 불가.
체계적 ablation(A: rainfall, B: dropout)으로 단일 원인 확정.

### 4. 동적 Query만이 유효한 Attention

고정 Query(step_queries) Cross-Attention: weight 균등 → 무효.
동적 Query(decoder_hidden_t) Bahdanau: recency bias이나 0.6%p 기여.

### 5. 비선형 Decoder는 이 데이터에서 실패

ReLU를 포함한 비선형 decoder(MultiHead, StepEmbed)는 underfitting.
단순한 `Linear(128→1)` + LSTMCell 순차 디코딩이 최적.

---

## 산출물

### 분석 보고서

| 파일 | 내용 |
|------|------|
| `12model_benchmark_report.md` | Phase 1: 12개 모델 실험 전체 결과 |
| `seq2seq_attn_v2_analysis.md` | Phase 2: v2 결과 분석 (4개 그래프 포함) |
| `ablation_decoder_dropout_analysis.md` | Phase 3: Ablation B 상세 분석 |
| `seq2seq_attn_v1_v2_v3_comparison.md` | v1/v2/v3 비교 보고서 |

### 노트북

| 파일 | 모델 | 상태 |
|------|------|------|
| `07_seq2seq_attn_60d_v1.ipynb` | v1 (원본 Best) | 확정 |
| `08_seq2seq_attn_60d_v2_layernorm.ipynb` | v2 (Revised) | 확정 (성능 하락) |
| `18_ablation_rainfall_restored.ipynb` | Ablation A | 확정 (rainfall ≠ 원인) |
| `19_ablation_decoder_dropout_removed.ipynb` | Ablation B = **v3 (최종)** | **확정** |
| `06_seq2seq_no_attn_60d.ipynb` | Seq2Seq (2위) | 확정 |
| `09_residual_fc_45d.ipynb` | Residual 45d (3위) | 확정 |
| `10_residual_fc_60d.ipynb` | Residual 60d (4위) | 확정 |
| `14_step_attn_mid_60d.ipynb` | Cross-Attn 8H (5위) | 확정 |
| `seq2seq_attn_flow_prediction.ipynb` | Baseline | 참고 (비표준) |
| `12_step_attn_base.ipynb` | Cross-Attn 4H | 확정 |
| `13_step_attn_mid.ipynb` | Cross-Attn 8H 45d | 확정 |
| `15_step_attn_high.ipynb` | Cross-Attn 8H + Huber | 참고 (비표준) |
| `16_step_embed_no_attn_60d.ipynb` | Step Embedding | 확정 (실패) |
| `11_multihead_attn_fc_60d.ipynb` | MultiHead FC | 확정 (실패) |
| `17_autoreg_prev_pred_feedback.ipynb` | Autoreg Attention | 확정 (실패) |
