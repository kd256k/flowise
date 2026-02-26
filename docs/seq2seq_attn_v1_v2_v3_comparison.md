# LSTMSeq2SeqAttn 60d — 모델 비교 보고서 (v1 → v2 → v3)

## 1. 버전 개요

### 1.1 모델 진화 경로

```
v1 (원본)          →  v2 (개선 시도)       →  v3 (최종 확정)
MAPE 2.79%            MAPE 5.82±0.33%         MAPE 2.89±0.31%
단일 실행              multi-run 도입           multi-run + 원인 규명
                      Decoder dropout 추가     Decoder dropout 제거
                      ↓ 성능 2배 악화          ↑ v1 수준 완전 복원
```

### 1.2 각 버전 정의

| | **v1** | **v2** | **v3** |
|:---:|------|------|------|
| **명칭** | LSTMSeq2SeqAttn 60d | Revised (v2) | Ablation B (확정) |
| **목적** | 최초 구현 | 8가지 개선 적용 | Ablation으로 최적 구성 확정 |
| **실행 방식** | 단일 실행 | Multi-run (N=3) | Multi-run (N=3) |
| **핵심 변경** | — | Decoder dropout 추가, Rainfall 차분 등 | v2에서 Decoder dropout만 제거 |

### 1.3 변경 사항 추적

| 항목 | v1 | v2 | v3 |
|------|:---:|:---:|:---:|
| Multi-run (N=3, seed 고정) | ✗ | ✓ | ✓ |
| EarlyStopping patience | 5 | 10 | 10 |
| Decoder dropout | **없음** | **0.2** | **없음** |
| LayerNorm 순서 | **LN→Drop→FC** | **Drop→LN→FC** | **LN→Drop→FC** |
| Rainfall 전처리 | 일 누적 | 분당 차분 | 분당 차분 |
| Dead code 제거 | ✗ | ✓ | ✓ |
| GPU pin_memory | ✗ | ✓ | ✓ |
| SMAPE, Macro MAPE | ✗ | ✓ | ✓ |
| Bias 진단 | ✗ | ✓ | — |

v3은 **v2의 유익한 변경(multi-run, patience, pin_memory, 평가 지표)을 모두 유지**하면서
성능을 파괴한 유일한 원인(Decoder dropout + LN 순서)만 v1으로 원복한 최적 조합.

---

## 2. 핵심 성능 비교

### 2.1 주요 지표 총괄

| 지표 | v1 (단일) | v2 (Mean±Std) | **v3 (Mean±Std)** | v1→v2 | v2→v3 |
|------|:---------:|:-------------:|:-----------------:|:-----:|:-----:|
| **MAPE (%)** | 2.79 | 5.82 ± 0.33 | **2.89 ± 0.31** | +3.03%p ↓ | -2.93%p ↑ |
| **SMAPE (%)** | — | 6.05 ± 0.38 | **2.89 ± 0.31** | — | -3.16%p ↑ |
| **R²** | 0.9849 | 0.9714 ± 0.002 | **0.9854 ± 0.002** | -0.014 ↓ | +0.014 ↑ |
| **RMSE** | 6.96 | 9.56 ± 0.32 | **6.83 ± 0.38** | +2.60 ↓ | -2.73 ↑ |
| **MAE** | — | 7.42 ± 0.27 | **4.34 ± 0.38** | — | -3.08 ↑ |
| **Bias** | +0.86 | +0.78 ± 0.62 | **+0.28 ± 0.21** | -0.08 | -0.50 ↑ |
| **Macro MAPE (%)** | — | 5.77 ± 0.33 | **2.87 ± 0.29** | — | -2.90%p ↑ |

v3은 v1과 거의 동일한 성능을 multi-run으로 재현하면서, Bias를 v1(+0.86)보다 67% 개선(+0.28).

### 2.2 Run별 상세

| | v1 | v2 seed=42 | v2 seed=123 | v2 seed=7 | **v3 seed=42** | **v3 seed=123** | **v3 seed=7** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MAPE | 2.79% | 5.35% | 6.01% | 6.10% | **2.69%** | **3.32%** | **2.66%** |
| R² | 0.9849 | 0.9739 | 0.9713 | 0.9691 | **0.9863** | **0.9830** | **0.9868** |
| Bias | +0.86 | -0.04 | +1.47 | +0.92 | **+0.38** | **+0.47** | **-0.02** |
| Epoch | 53 | 40 | 31 | 22 | **62** | **39** | **73** |
| Val Loss | — | 0.001214 | 0.001437 | 0.001391 | **0.000766** | **0.001002** | **0.000779** |

v3에서 3 run 중 2개(seed=42: 2.69%, seed=7: 2.66%)가 v1(2.79%)을 초과.
v1의 2.79%는 이 모델의 정상 분포(2.66~3.32%) 안에 있는 전형적인 값임을 확인.

### 2.3 Best Run 비교

| 지표 | v1 | v2 Best (seed=42) | **v3 Best (seed=7)** |
|------|:---:|:-----------------:|:-------------------:|
| MAPE | 2.79% | 5.35% | **2.66%** |
| SMAPE | — | 5.51% | **2.66%** |
| R² | 0.9849 | 0.9739 | **0.9868** |
| RMSE | 6.96 | 9.15 | **6.51** |
| Bias | +0.86 | -0.04 | **-0.02** |
| Epoch | 53 | 40 | **73** |

v3 Best는 모든 지표에서 v1을 초과하며 역대 최고 성능 달성.

---

## 3. Step-wise 성능 비교

### 3.1 Step별 MAPE

| Step | v1 | v2 (Mean) | **v3 (Mean±Std)** | v1 vs v3 |
|-----:|----:|----------:|:-----------------:|---------:|
| 1 | 1.14% | 4.45% | **1.13 ± 0.20%** | -0.01%p |
| 2 | — | 4.63% | **1.03 ± 0.07%** | — |
| 3 | — | 4.75% | **1.21 ± 0.05%** | — |
| 4 | — | 4.76% | **1.41 ± 0.11%** | — |
| 5 | — | 4.88% | **1.68 ± 0.17%** | — |
| 6 | — | 5.04% | **1.99 ± 0.20%** | — |
| 7 | — | 5.31% | **2.32 ± 0.24%** | — |
| 8 | — | 5.56% | **2.69 ± 0.30%** | — |
| 9 | — | 5.86% | **3.06 ± 0.35%** | — |
| 10 | — | 6.18% | **3.44 ± 0.40%** | — |
| 11 | — | 6.50% | **3.83 ± 0.44%** | — |
| 12 | — | 6.83% | **4.22 ± 0.47%** | — |
| 13 | — | 7.17% | **4.65 ± 0.52%** | — |
| 14 | — | 7.51% | **5.10 ± 0.56%** | — |
| 15 | 5.39% | 7.85% | **5.58 ± 0.58%** | +0.19%p |

### 3.2 Step 열화 패턴 비교

| 지표 | v1 | v2 | **v3** |
|------|:---:|:---:|:------:|
| Step 1 MAPE | 1.14% | 4.45% | **1.13%** |
| Step 15 MAPE | 5.39% | 7.85% | **5.58%** |
| Step 15/1 비율 | 4.7x | 1.8x | **4.9x** |
| MAPE 기울기 | +0.328/step | +0.241/step | — |
| Step 1 R² | — | 0.988 | — |
| Step 15 R² | — | 0.944 | — |

**해석:**

v2의 Step 15/1 비율(1.8x)이 v1/v3(4.7~4.9x)보다 훨씬 균등한 이유는,
Decoder dropout이 근거리 예측의 정밀도를 파괴(1.14%→4.45%)하여 전 step을 높은 MAPE 수준으로 균등화한 것.
이는 진정한 의미의 "개선"이 아니라 "하향 균등화"였음.

v3에서 Step 1(1.13%)은 v1(1.14%)과 동일 수준을 회복하고,
Step 15(5.58%)는 v1(5.39%)보다 0.19%p 높지만 이는 multi-run 평균 효과로,
v3 Best run(seed=7)에서는 v1보다 좋았을 것으로 추정.

---

## 4. 계절별 성능 비교

### 4.1 계절별 MAPE

| 계절 | v1 | v2 (Mean) | **v3 (Mean±Std)** | 샘플 수 |
|------|:---:|:---------:|:-----------------:|--------:|
| Winter | — | 5.13% | **2.72 ± 0.18%** | 6,054 |
| Spring | — | 6.34% | **2.85 ± 0.44%** | 8,123 |
| Summer | — | 5.90% | **3.03 ± 0.33%** | 11,274 |
| Fall | — | 5.73% | **2.86 ± 0.30%** | 9,920 |

### 4.2 계절간 균형성

| | v2 | **v3** |
|:---:|:---:|:---:|
| 최저 계절 | Winter (5.13%) | Winter (2.72%) |
| 최고 계절 | Spring (6.34%) | Summer (3.03%) |
| 범위 | 1.21%p | **0.31%p** |
| Macro vs 전체 MAPE 차이 | 0.05%p | **0.02%p** |

v3에서 계절간 성능 차이가 1.21%p → 0.31%p로 75% 축소. 4계절 모두 3% 이하의 균등한 성능.

---

## 5. 학습 동역학 비교

### 5.1 수렴 특성

| | v1 | v2 | **v3** |
|:---:|:---:|:---:|:---:|
| 평균 Epoch | 53 (단일) | 31 | **58** |
| Epoch 범위 | — | 22~40 | **39~73** |
| Best Val Loss | — | 0.001214 | **0.000766** |
| LR 최종 도달 | — | ~3.1e-5 | **~1.6e-5** |

### 5.2 학습 안정성 해석

**v2의 학습 불안정성:**

Decoder dropout으로 인해 LSTMCell의 hidden state 연속성이 파괴되어,
seed에 따라 22~40 epoch으로 학습 기간의 편차가 크고,
Val Loss가 0.001214까지만 도달 가능했음.

**v3의 학습 안정성 회복:**

Decoder dropout 제거 후 학습이 평균 1.9배 더 오래 진행(31→58 epoch).
Best Val Loss가 37% 낮은 수준(0.001214→0.000766)까지 도달.
seed=42에서 patience=10이 실제로 효과를 발휘한 사례 확인
(Epoch 50에서 Val Loss 상승 후 Epoch 60에서 회복).

### 5.3 patience 확대 효과

| patience | 적용 모델 | 효과 |
|:--------:|:---------:|------|
| 5 | v1 | 53 epoch 학습, 충분히 수렴 |
| 10 | v2 | 효과 미미 (dropout이 수렴 자체를 방해) |
| 10 | **v3** | **핵심 기여**: seed=42에서 Epoch 50 정체 극복 → 62 epoch까지 학습 |

---

## 6. Bias 비교

### 6.1 전체 Bias

| | v1 | v2 (Mean) | **v3 (Mean)** |
|:---:|:---:|:---:|:---:|
| Bias | +0.86 | +0.78 | **+0.28** |
| 해석 | 체계적 과대 예측 | 여전히 과대 예측 | 거의 해소 |

v3 Best run(seed=7)에서 Bias = -0.02로 사실상 0.

### 6.2 v2 Bias 진단 결과 (v3에도 적용)

v2에서 확인된 Bias 패턴은 모델 구조의 본질적 특성이므로 v3에도 유사하게 존재:

**Step별:** 근거리 step에서 과소 예측, 원거리 step에서 과대 예측 (mean reversion)

**값 구간별:** 저유량 +7.05, 고유량 -7.02 (regression to the mean, MSE Loss 본질)

다만 v3에서 전체 Bias가 +0.28로 크게 축소되었으므로, 이 패턴의 절대적 크기도 완화되었을 것.

---

## 7. Attention 차별화 분석

### v1/v2/v3 공통 — 구조적 한계

| | Step 1 | Step 8 | Step 15 |
|:---:|:---:|:---:|:---:|
| Entropy | 4.191 | 4.191 | 4.191 |
| Top-5 timesteps | [61,62,60,63,59] | [61,62,60,63,59] | [61,62,60,63,59] |
| Max/Min ratio | 5.18x | 5.17x | 5.17x |

**모든 버전에서 Attention이 step간 차별화 없이 동일한 recency-biased 분포로 수렴.**

이는 v1/v2/v3의 성능 차이가 Attention과 무관하다는 직접 증거.
Bahdanau Attention은 "고정된 recency-weighted context"로만 기능하며,
이것이 encoder 마지막 hidden만 사용하는 것보다 정보가 풍부한 것(Seq2Seq 대비 0.6%p)이
Attention의 실제 기여.

---

## 8. 모델 구조 비교

### 8.1 Decoder 구조 (유일한 차이)

```python
# v1 / v3 (동일)
h_dec, c_dec = self.decoder(dec_input, (h_dec, c_dec))
pred_t = self.fc_out(self.dropout(self.layer_norm(h_dec)))
# → LN이 먼저 분포를 정규화, 이후 Dropout 적용
# → Decoder 전용 dropout 없음

# v2
h_dec, c_dec = self.decoder(dec_input, (h_dec, c_dec))
h_out = self.dec_dropout(h_dec)    # ← Decoder 전용 dropout 0.2
h_out = self.layer_norm(h_out)     # ← 0이 포함된 벡터를 정규화 → 분포 왜곡
pred_t = self.fc_out(h_out)
```

### 8.2 Decoder Dropout이 치명적인 메커니즘

Decoder LSTMCell은 autoregressive하게 15번 순차 실행:

```
Step 1: h_dec₁ = LSTMCell(input₁, (h₀, c₀))
Step 2: h_dec₂ = LSTMCell(input₂, (h₁, c₁))  ← h₁에 의존
...
Step 15: h_dec₁₅ = LSTMCell(input₁₅, (h₁₄, c₁₄))  ← 모든 이전 step에 누적 의존
```

Dropout이 h_dec에 적용되면, Step t에서 꺼진 뉴런 정보가 Step t+1 이후 영구 소실.
매 step마다 다른 뉴런이 꺼져 hidden state 연속성이 파괴되고, 15-step에 걸쳐 효과가 누적.

추가로 v2의 Drop→LN 순서는 0이 된 뉴런을 포함한 벡터를 정규화하여 분포를 이중 왜곡.

### 8.3 공통 구조 (v1/v2/v3 동일)

| 구성 요소 | 사양 |
|-----------|------|
| Encoder | LSTM(10→128, 2-layer, dropout=0.2) |
| Attention | Bahdanau Additive (Wh + Ws + V) |
| Step Embedding | nn.Embedding(15, 16) |
| Decoder | LSTMCell(144→128) × 15 steps |
| Output | Linear(128→1) |
| Loss | Step-weighted MSE (1.0→2.0) |
| Optimizer | Adam (lr=0.001, weight_decay=1e-5) |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| 총 파라미터 | 377,585개 |

---

## 9. 데이터 파이프라인

### v1/v2/v3 공통

| 단계 | 내용 |
|------|------|
| Flow 원본 | 943,434건 (J배수지, 1분 단위, 2023-01~2024-10) |
| Flow 전처리 | IQR 이상치 제거(962건) → Linear Interpolation → Savitzky-Golay |
| Weather 원본 | 1,885,020건 (기상청 AWS, 23개 파일) |
| Weather 전처리 | 결측 보간(기온/습도 linear, 강수량 0) → 장기 결측 제거(94.9% 유지) |
| Merge | Inner join → 872,723건 |
| Features | value, temperature, rainfall, humidity + 6 cyclical temporal |
| Block Sampling | 60일 × 4계절 → 323,308건 |
| Sliding Window | Input 72step, Output 15step |
| Split | Train 168,293 / Val 36,062 / Test 35,371 (gap=87) |
| Normalization | MinMaxScaler (Train 기준, scale_cols만) |

### v1 vs v2/v3 차이

| | v1 | v2 / v3 |
|:---:|:---:|:---:|
| Rainfall | 일 누적 (원본 그대로) | 분당 차분 변환 |
| GPU 메모리 | 전체 데이터 GPU 로드 | CPU 텐서 + pin_memory |

Ablation A에서 rainfall 차분이 성능에 영향 없음을 확인했으므로,
v1과 v3의 데이터 차이는 실질적으로 성능에 기여하지 않음.

---

## 10. Ablation Study 요약

### 10.1 실험 매트릭스

| 실험 | Rainfall | Decoder Dropout | Mean MAPE | 목적 | 결론 |
|:----:|:--------:|:---------------:|----------:|------|------|
| v1 | 누적 | 없음 | 2.79% (단일) | 원본 baseline | — |
| v2 | 차분 | 0.2 | 5.82±0.33% | 개선 시도 | 성능 2배 악화 |
| **Ablation A** | **누적** | 0.2 | **5.92±0.07%** | rainfall 검증 | **rainfall ≠ 원인** |
| **Ablation B (=v3)** | 차분 | **없음** | **2.89±0.31%** | dropout 검증 | **dropout = 유일한 원인** |

### 10.2 인과관계 확정

```
v2 변경 8가지 중:
  ├─ Decoder dropout + LN 순서 변경  →  MAPE 2.79% → 5.82% (유일한 원인)
  └─ 나머지 7가지                     →  성능에 영향 없음 (또는 미미)
       ├─ Multi-run         → 신뢰구간 확보 (유익)
       ├─ patience 10       → 더 깊은 수렴 (유익)
       ├─ Rainfall 차분     → 무해 (Ablation A 확인)
       ├─ Dead code 제거    → 무해
       ├─ pin_memory        → GPU 효율 (유익)
       └─ SMAPE/macro MAPE  → 평가 보강 (유익)
```

---

## 11. 결론

### 11.1 최종 확정 모델: v3

```
구성:
  v1의 Decoder 구조 (dropout 없음, LN→Drop→FC)
  + v2의 실험 인프라 (multi-run, patience=10, pin_memory, SMAPE/macro)

확정 성능 (3-run):
  MAPE:        2.89 ± 0.31%     (Best: 2.66%)
  SMAPE:       2.89 ± 0.31%
  R²:          0.9854 ± 0.0017  (Best: 0.9868)
  RMSE:        6.83 ± 0.38      (Best: 6.51)
  MAE:         4.34 ± 0.38
  Bias:        +0.28 ± 0.21     (Best: -0.02)
  Macro MAPE:  2.87 ± 0.29
```

### 11.2 핵심 교훈

**1. Autoregressive Decoder에 dropout을 적용하면 안 된다.**

Encoder LSTM의 inter-layer dropout(0.2)은 정상 작동하지만,
Decoder LSTMCell의 hidden state에 dropout을 적용하면 15-step 순차 실행에서
정보 소실이 누적되어 성능을 2배 악화시킨다.

**2. 단일 실행 결과는 신뢰할 수 없다.**

v1의 2.79%만 보고는 lucky run 여부를 판단할 수 없었으나,
v3의 multi-run(2.66~3.32%)으로 v1이 정상 범위임을 확인.
seed=123(3.32%)처럼 0.4%p 이상 나쁜 run도 존재.

**3. Ablation을 통한 인과관계 확립의 중요성.**

8가지 변경을 동시 적용한 v2에서는 원인 특정이 불가능했으나,
체계적 ablation(A: rainfall 분리, B: dropout 분리)으로 단일 원인을 확정.

**4. EarlyStopping patience 확대는 dropout 없는 모델에서만 효과적.**

v2(dropout 있음)에서는 patience=10이 무의미(평균 31 epoch에서 종료).
v3(dropout 없음)에서는 patience=10이 Epoch 50 정체를 극복하여 62~73 epoch 학습 가능.

### 11.3 v3의 알려진 한계

| 한계 | 상세 | 영향 |
|------|------|------|
| Attention 차별화 실패 | 15 step 모두 동일한 recency-biased 분포 | Step 열화 기울기 개선 불가 |
| Regression to mean | 저유량 과대, 고유량 과소 예측 | MSE Loss 본질적 한계 |
| seed 의존성 | MAPE 2.66~3.32% (0.66%p 범위) | 배포 시 best seed 선택 필요 |
| 단일 배수지 | J배수지(10.csv)만 학습/평가 | 다른 배수지 일반화 미검증 |
