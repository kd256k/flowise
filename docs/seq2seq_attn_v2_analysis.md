# LSTMSeq2SeqAttn 60d v2 — 실험 결과 분석 보고서

## 1. 실험 개요

### 목적
기존 v1 (MAPE 2.79%, 단일 실행)에 8가지 개선 사항을 적용한 v2의 성능을 평가하고,
multi-run을 통해 모델의 실제 성능 분포를 확인한다.

### v2 변경 사항 요약

| # | 변경 | 내용 |
|:-:|------|------|
| 1 | Multi-run + seed 고정 | N=3, seeds=[42, 123, 7] |
| 2 | Bias 진단 | Scaler 범위, 계절·Step·값 구간별 분해 |
| 3 | EarlyStopping patience | 5 → 10 |
| 4 | Decoder dropout | 0.2 추가, Drop→LN→FC 순서 |
| 5 | Rainfall 차분 | 일 누적 → 분당 변화량 |
| 6 | Dead code 제거 | `_bahdanau_attention()` 삭제 |
| 7 | GPU 메모리 | CPU 텐서 + pin_memory |
| 8 | 평가 지표 | SMAPE, macro-avg MAPE 추가 |

---

## 2. Multi-Run 결과 (3회 실행)

### 2.1 Run별 성능 비교

| Run | Seed | MAPE | SMAPE | R² | Bias | Epoch |
|:---:|:----:|-----:|------:|---:|-----:|------:|
| 1 | 42 | **5.35%** | **5.51%** | **0.9739** | **-0.04** | 40 |
| 2 | 123 | 6.01% | 6.28% | 0.9713 | +1.47 | 31 |
| 3 | 7 | 6.10% | 6.36% | 0.9691 | +0.92 | 22 |

### 2.2 통계 요약

| 지표 | Mean | Std | Min | Max |
|------|-----:|----:|----:|----:|
| MAPE (%) | 5.82 | 0.33 | 5.35 | 6.10 |
| SMAPE (%) | 6.05 | 0.38 | 5.51 | 6.36 |
| R² | 0.9714 | 0.0019 | 0.9691 | 0.9739 |
| RMSE | 9.56 | 0.32 | 9.15 | 9.94 |
| MAE | 7.42 | 0.27 | 7.06 | 7.71 |
| Bias | +0.78 | 0.62 | -0.04 | +1.47 |
| Macro MAPE (%) | 5.77 | 0.33 | 5.31 | 6.05 |

### 2.3 Run간 차이 분석

3회 실행 간 MAPE 범위가 5.35%~6.10% (0.75%p)로, seed에 따른 학습 variance가 상당히 크다.

**seed=42가 best인 이유:**
40 epoch까지 학습(다른 run 대비 최장)하며 충분히 수렴. Val Loss 0.001214로 3 run 중 최저. LR scheduler가 0.001→0.0005→0.000125→0.000031로 4단계 감소하며 fine-grained 최적화를 수행.

**seed=7이 worst인 이유:**
22 epoch에서 조기 종료. LR이 0.00025까지만 감소하고 수렴 기회를 충분히 받지 못함. patience=10이었지만 val loss가 빠르게 정체.

이 결과는 **v1의 MAPE 2.79%가 특별히 좋은 seed와 충분한 학습(53 epoch)의 조합**이었을 가능성을 시사하며, 모델의 실제 성능 분포는 5~6% 대에 있다.

---

## 3. v1 vs v2 비교 분석

### 3.1 핵심 지표 비교

| 지표 | v1 (단일) | v2 Best (seed=42) | v2 Mean±Std |
|------|:---------:|:-----------------:|:-----------:|
| MAPE | **2.79%** | 5.35% | 5.82±0.33% |
| R² | **0.9849** | 0.9739 | 0.9714±0.002 |
| RMSE | **6.96** | 9.15 | 9.56±0.32 |
| Bias | +0.86 | **-0.04** | +0.78±0.62 |
| Step 1 MAPE | **1.14%** | 3.98% | 4.45±0.34% |
| Step 15 MAPE | **5.39%** | 7.28% | 7.85±0.40% |
| Step 기울기 | +0.328/step | **+0.236/step** | — |
| Step 15/1 비율 | 4.7x | **1.8x** | — |
| Epoch | 53 | 40 | 31±7 |

### 3.2 해석

전체 MAPE는 v1 대비 약 2배 악화되었지만, 두 가지 긍정적 변화가 있다:

**Step 열화 기울기 개선**: v1의 +0.328%/step → v2의 +0.236%/step로 27% 감소. Step 15/1 비율도 4.7x → 1.8x로 크게 개선되어, 원거리 예측의 상대적 안정성이 향상됨. 이는 Decoder dropout이 근거리 예측의 과적합을 억제하면서 전체적으로 균등한 성능을 유도한 결과.

**Best Run의 Bias 해소**: seed=42에서 Bias=-0.04로 거의 0에 가까워, 체계적 편향이 사라짐.

다만 이 트레이드오프는 전체 MAPE 기준으로는 명확한 손해이며, Step 1의 MAPE가 1.14%→3.98%로 3.5배 악화된 것이 전체 성능 하락의 주요 원인.

---

## 4. 그래프 분석

### 4.1 Training History (Image 1)

**좌측 (Linear Scale):** 3개 run의 학습 곡선이 비교됨.
seed=42 (파란 실선)가 가장 깊이 수렴하며, Train Loss ~0.0009, Val Loss ~0.0012 수준에 도달.
seed=123/7 (연한 색)은 Val Loss가 seed=42보다 높은 수준에서 정체되어 조기 종료.

**우측 (Log Scale):** 핵심 패턴이 명확히 드러남.
- seed=42: epoch 25 이후 Train/Val 모두 10⁻³ 아래로 안정적 수렴
- seed=123: Val Loss 진동 폭이 크고, epoch 20~30 구간에서 불안정
- seed=7: Val Loss가 가장 높은 수준에서 빠르게 정체 → 22 epoch에서 종료

**시사점:** Decoder dropout 추가로 학습 난이도가 높아졌으며, 좋은 초기화(seed)를 만나야 깊이 수렴할 수 있는 구조가 됨. v1에서 dropout 없이 53 epoch까지 안정적으로 학습한 것과 대조적.

### 4.2 Step-wise MAPE with Variance (Image 2)

**패턴:** Step 1(4.5±0.3%) → Step 15(7.9±0.4%)로 단조 증가. Mean 5.82%.

**주요 관찰:**
- Step 1~5 구간: 4.5~4.9%로 비교적 완만한 증가 (약 +0.1%/step)
- Step 6~15 구간: 5.0→7.9%로 가속적 증가 (약 +0.3%/step)
- 전환점이 Step 6 부근에 존재 — Decoder LSTMCell이 ~6 step까지는 context를 유지하지만, 이후 정보가 감쇠

**Error bar (std):** 모든 step에서 ±0.3~0.4%로 비교적 균일. step이 멀어져도 seed 의존성이 비례적으로 증가하지는 않음. 이는 variance의 주 원인이 decoder가 아니라 encoder (LSTM 초기화)에 있음을 시사.

**v1 대비:**
v1은 Step 1: 1.14%, Step 15: 5.39%로 출발점이 훨씬 낮았지만 비율은 4.7배.
v2는 출발점이 높은 대신 비율이 1.8배로 균등. Decoder dropout이 "전 step 균등화" 효과를 낸 것으로 해석.

### 4.3 Step Degradation Analysis (Image 3)

4개 지표(MAPE, RMSE, MAE, R²) × 5개 계열(All + 4계절)의 step별 추세.

**MAPE vs Step (좌상):**
- 전체 기울기: +0.241%/step (v1의 +0.328% 대비 27% 개선)
- Winter(파란 점선)가 초반 가장 낮고(Step 1: 3.2%) 후반에 급증 → 기울기 +0.309/step으로 최대
- Spring(초록 점선)이 전 구간에서 가장 높음 (Mean 6.34%)
- Summer/Fall은 전체 평균과 유사

**RMSE vs Step (우상):**
- 기울기 +0.508/step — 절대 오차가 step마다 ~0.5 단위씩 증가
- Winter가 Step 13 이후 급격히 상승하는 특이 패턴 (겨울 고유량 구간의 큰 절대 오차)

**MAE vs Step (좌하):**
- Step 1: ~5.1 → Step 15: ~10.1로 약 2배 증가
- 계절 간 편차가 MAPE보다 작음 — MAE는 스케일 의존적이므로 유량 수준이 비슷한 계절끼리 수렴

**R² vs Step (우하):**
- Step 1: 0.988 → Step 15: 0.944로 감소. 기울기 -0.003/step
- Step 15에서도 R²=0.94 이상 유지 — 실용적 예측 성능은 확보
- Winter가 Step 15에서 0.925로 유독 낮음 (겨울 수요 패턴의 높은 변동성 반영)

**핵심 발견:** 계절별 열화 패턴이 v1 보고서(+0.328 기울기 균일)와 달리 v2에서는 Winter(+0.309)와 Spring(+0.217) 간 차이가 벌어짐. Decoder dropout이 계절별로 다른 영향을 미치고 있음.

### 4.4 Bahdanau Attention Heatmap (Image 4)

**좌측 (Heatmap):**
15 decoder step(y축) × 72 encoder timestep(x축)의 attention weight 분포.
색상 패턴이 모든 step에서 완전히 동일 — 최근 timestep(60~72)에 높은 가중치, 초반(1~20)에 낮은 가중치.
step 간 차별화가 전혀 없는 recency bias 고정 패턴.

**우측 (Distribution):**
Step 1, 8, 15의 attention distribution이 **완전히 겹침** (3개 선이 하나로 보임).
범례의 한글이 깨져 □으로 표시되는 폰트 문제가 있지만, 빨간 선(Step 15)만 보임 = 3개가 동일.

**정량 확인:**

| | Step 1 | Step 8 | Step 15 |
|:---:|:---:|:---:|:---:|
| Max weight | 0.0180 | 0.0180 | 0.0180 |
| Max timestep | 62 | 62 | 62 |
| Min weight | 0.0035 | 0.0035 | 0.0035 |
| Max/Min ratio | 5.18x | 5.17x | 5.17x |
| Top-5 | [61,62,60,63,59] | [61,62,60,63,59] | [61,62,60,63,59] |
| Entropy | 4.191 | 4.191 | 4.191 |
| Uniform entropy | 4.277 | 4.277 | 4.277 |

Entropy 4.191 (uniform=4.277)로 약간의 집중이 있지만, step 간 차이가 소수점 3자리까지 동일.

**원인 분석:**
Bahdanau Attention의 Query = `W_h(h_dec)` 인데, h_dec가 step간 거의 변하지 않음.
step_embedding(16-dim)이 Decoder LSTMCell 입력에 들어가지만, context(128-dim) 대비 비중이 11%에 불과.
LSTMCell의 gate mechanism이 이 작은 차이를 더 억제하여 h_dec의 step간 변화가 float 정밀도 수준.

**결론:** Attention이 "recency-weighted average"로 고정되어 있으며, 이는 v1에서도 동일.
MAPE 2.79%(v1) vs 5.82%(v2)의 성능 차이는 Attention과 무관하고, Decoder LSTMCell의 순차 디코딩 자체와 학습 수렴 깊이의 차이.

---

## 5. Bias 진단 결과

### 5.1 MinMaxScaler 범위

| | Train | Test |
|:---:|:---:|:---:|
| Value 범위 | [0.00, 310.26] | [55.88, 321.34] |
| 범위 밖 (above) | — | 91건 (0.02%) |
| 범위 밖 (below) | — | 0건 |

Train max(310.26) 초과 값이 91건 존재하지만 0.02%로 무시 가능. Scaler 범위 불일치는 Bias의 주 원인이 아님.

### 5.2 계절별 Bias

| 계절 | Bias | 상대 Bias | 평균 실측 | 샘플 수 |
|------|-----:|----------:|----------:|--------:|
| Winter | -0.75 | -0.44% | 170.2 | 6,054 |
| Spring | **+1.42** | **+1.01%** | 139.8 | 8,123 |
| Summer | -0.26 | -0.17% | 151.8 | 11,274 |
| Fall | -0.55 | -0.35% | 156.0 | 9,920 |

Spring만 양의 Bias(과대 예측)이고 나머지는 음의 Bias(과소 예측). Spring의 평균 실측이 139.8로 가장 낮은데 과대 예측한다는 것은, 모델이 저유량 구간에서 높게 예측하는 경향이 있음을 시사.

### 5.3 Step별 Bias 패턴

Step 1~7: 음의 Bias (과소 예측, -0.87 → -0.17)
Step 8: Bias ≈ 0 (전환점)
Step 9~15: 양의 Bias (과대 예측, +0.10 → +1.02)

**해석:** 근거리 step에서는 실제보다 낮게, 원거리 step에서는 실제보다 높게 예측.
이는 Decoder가 시간이 지남에 따라 "평균 회귀(mean reversion)" 경향을 학습한 결과.
유량이 급변하는 시점에서 근거리는 변화를 따라가되 약간 보수적이고, 원거리는 장기 평균 쪽으로 회귀.

### 5.4 값 구간별 Bias

| 구간 | 범위 | Bias | 샘플 수 |
|------|------|-----:|--------:|
| Q1 (저유량) | 55.9~112.4 | **+7.05** | 132,651 |
| Q2-Q3 (중간) | 112.4~193.1 | **-0.09** | 265,301 |
| Q4 (고유량) | 193.1~321.3 | **-7.02** | 132,643 |

**핵심 발견:** 저유량에서 +7.05, 고유량에서 -7.02로 거의 대칭적. 모델이 중간값(~150) 쪽으로 예측을 수축하는 "regression to the mean" 패턴. 이는 MSE Loss의 본질적 특성(극단값보다 중심값 예측이 loss를 줄임)에 기인.

---

## 6. Step 열화 정량 분석

### 6.1 전체 + 계절별 기울기

| 계열 | MAPE 기울기 | RMSE 기울기 | R² 기울기 | Bias 기울기 |
|------|:----------:|:----------:|:--------:|:----------:|
| All | +0.241/step | +0.508/step | -0.003/step | +0.152/step |
| Winter | +0.309/step | +0.578/step | -0.004/step | +0.364/step |
| Spring | +0.217/step | +0.452/step | -0.003/step | +0.131/step |
| Summer | +0.233/step | +0.500/step | -0.003/step | +0.097/step |
| Fall | +0.229/step | +0.514/step | -0.003/step | +0.102/step |

Winter의 MAPE 기울기(+0.309)가 Spring(+0.217)보다 42% 높음. Winter는 Step 1에서 3.21%로 가장 낮지만 Step 15에서 7.25%로 급증 → 겨울 수요의 급격한 일중 변동이 원거리 예측을 어렵게 만듦.

### 6.2 Step 1 vs Step 15

| 계열 | Step 1 MAPE | Step 15 MAPE | 비율 | Step 1 R² | Step 15 R² |
|------|:----------:|:-----------:|:---:|:---------:|:----------:|
| All | 3.98% | 7.28% | 1.8x | 0.9880 | 0.9440 |
| Winter | 3.21% | 7.25% | 2.3x | 0.9852 | 0.9247 |
| Spring | 4.28% | 7.40% | 1.7x | 0.9888 | 0.9484 |
| Summer | 4.03% | 7.23% | 1.8x | 0.9882 | 0.9456 |
| Fall | 4.12% | 7.26% | 1.8x | 0.9876 | 0.9442 |

Step 15에서도 전체 R²=0.944으로 높은 설명력 유지. Winter만 Step 15 R²=0.925로 다소 낮지만 실용적 수준.

### 6.3 Step-weighted Loss 효과

| 지표 | 값 |
|------|-----|
| Step 1 weight | 1.0 |
| Step 15 weight | 2.0 |
| Step 15/Step 1 MAPE 비율 | 1.8x |
| MAPE 증가율 | +0.236%/step |

Step 15에 2배 가중했음에도 Step 15 MAPE가 Step 1의 1.8배. Weight 2배 → MAPE 비율 1.8배로, Loss 가중치가 성능 비율에 거의 비례하게 작용하고 있음.

---

## 7. Attention 차별화 실패 분석

### 7.1 문제 정의

Bahdanau Attention의 설계 의도: "Step 1은 최근 timestep에 집중, Step 15는 더 넓은 범위를 참조"
실제 결과: 모든 step이 동일한 recency-biased 분포로 수렴

### 7.2 근본 원인

```
Decoder loop 내 정보 흐름:
  h_dec → Query = W_h(h_dec) → Attention Score → context
  context + step_emb(16dim) → LSTMCell → h_dec (다음 step)
```

순환 의존성: context가 h_dec에 의존하고, h_dec가 context에 의존하는 circular dependency.
Step 1에서 결정된 Attention 패턴이 이후 step에서도 거의 동일하게 재생산됨.

step_embedding(16-dim)이 유일한 step간 차별화 요소이지만, context(128-dim) 대비 11% 비중으로 LSTMCell gate에 의해 억제됨.

### 7.3 성능에 미치는 영향

Attention이 차별화되지 않았지만, "고정된 recency-weighted context"가 encoder 마지막 hidden만 사용하는 것보다는 정보가 풍부함.
v1 보고서에서 Seq2SeqAttn(2.79%) vs Seq2Seq(3.39%) = 0.6%p 차이는 이 "고정 context"의 기여.
실제 dynamic attention의 효과는 0에 가까움.

---

## 8. 기술적 이슈

### 8.1 한글 폰트 깨짐

Attention Distribution 그래프 범례에서 한글이 □으로 표시됨.
원인: matplotlib의 기본 폰트(DejaVu Sans)에 한글 글리프 미포함.

해결: `matplotlib.rc('font', family='NanumGothic')` 또는 범례를 영문으로 변경.

### 8.2 Test 경계 구간

| 계절 | 마지막 3일 | MAPE | vs 전체 | Bias |
|------|-----------|-----:|--------:|-----:|
| Winter | 01/25~01/28 | 4.54% | -0.81%p | -0.77 |
| Spring | 05/09~05/12 | 6.14% | +0.79%p | +1.78 |
| Summer | 08/24~08/27 | 5.29% | -0.06%p | -0.61 |
| Fall | 11/10~11/12 | 6.09% | +0.74%p | -0.79 |

Spring과 Fall의 경계 구간이 전체 평균보다 0.7~0.8%p 높음. 블록 끝부분의 데이터 특성 변화(계절 전환기)가 반영된 것으로 보임.

---

## 9. 결론

### 핵심 발견

1. **v2의 실제 성능 분포는 MAPE 5.82±0.33%** — v1의 2.79%는 재현되지 않음
2. **Step 열화 기울기 27% 개선** — Decoder dropout의 균등화 효과 (+0.328 → +0.241/step)
3. **Bias의 "regression to mean" 패턴** — 저유량 +7.05, 고유량 -7.02로 대칭적 수축
4. **Attention 차별화 완전 실패** — v1에서도 동일, 구조적 한계 확인
5. **seed 의존성 높음** — Best(5.35%) vs Worst(6.10%)로 0.75%p 차이

### 진행 중인 Ablation

v1→v2 성능 하락 원인 분리를 위한 ablation 실험 진행 중:

| 실험 | Rainfall | Dropout | MAPE (mean) | 결과 |
|:---:|:---:|:---:|:---:|:---:|
| v2 | 차분 | 있음 | 5.82% | baseline |
| A | 누적 (원복) | 있음 | 5.92% | **rainfall ≠ 원인** |
| B | 차분 | 없음 (원복) | *대기* | dropout 검증 |
| C | 누적 (원복) | 없음 (원복) | *대기* | 재현성 검증 |

Ablation A 결과로 rainfall 차분 변환은 성능 하락 원인이 아님이 확인됨. Ablation B·C 결과에 따라 dropout이 주 원인인지, v1 자체가 lucky run인지 최종 판단 예정.
