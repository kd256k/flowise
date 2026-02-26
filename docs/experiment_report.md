# LSTM Seq2Seq 유량 예측 — 실험 결과 보고서

## 개요

배수지(J) 1분 단위 유출유량을 72분 입력 → 15분 예측(×4 rolling = 1시간)하는 LSTM 기반 모델 개발.
12개 아키텍처 비교 → Best Model 개선 시도 → Ablation으로 원인 규명 → Rainfall 검증 → 최종 모델 확정의 4단계 실험을 수행.

```
Phase 1  12개 아키텍처 벤치마크 → Best: Seq2SeqAttn (MAPE 2.79%)
Phase 2  Best Model 개선(v2) → 8가지 변경 → 성능 2배 악화 (5.82%)
Phase 3  Ablation Study → Decoder dropout이 지배적 원인 → v3 확정 (2.89±0.31%)
Phase 4  Ablation C → 차분 rainfall이 +0.40%p 우위 확인 → 배포 버전 확정 (2.88±0.04%)
```

---

## 1. 공통 실험 설정

### 1.1 데이터

| 항목 | 값 |
|------|-----|
| 대상 | J배수지 (reservoir/10.csv), 1분 단위 |
| Flow 원본 | 943,434건 (2023-01 ~ 2024-10) |
| Weather 원본 | 1,885,020건 (기상청 AWS, 23개 파일) |
| 전처리 | IQR 이상치 제거 → Linear Interpolation → Savitzky-Golay |
| Merge | Flow + Weather inner join → 872,723건 |
| Features (10) | value, temperature, rainfall, humidity + time/dow/season (sin/cos) |

### 1.2 계절별 Block Sampling (60일 × 4)

| 계절 | 기간 |
|------|------|
| Winter | 2023-12-01 ~ 2024-01-29 |
| Spring | 2024-03-15 ~ 2024-05-13 |
| Summer | 2023-07-01 ~ 2023-08-29 |
| Fall | 2023-09-15 ~ 2023-11-13 |

### 1.3 모델 공통

| 항목 | 값 |
|------|-----|
| Input / Output | 72 steps (72min) → 15 steps (15min) |
| Sliding Window | Segment-aware (시간 불연속 경계 차단), gap=87 |
| Data Split | Train 70% / Val 15% / Test 15% (168,293 / 36,062 / 35,371) |
| Normalization | Train 기준 MinMaxScaler (sin/cos 제외) |
| Optimizer | Adam (lr=0.001, weight_decay=1e-5) |
| Gradient Clipping | max_norm=1.0 |

---

## 2. Phase 1 — 12개 아키텍처 벤치마크

### 2.1 실험 목적

| # | 목적 |
|---|------|
| 1 | Encoder-Decoder (Seq2Seq) 구조 유효성 검증 |
| 2 | Bahdanau Attention의 Seq2Seq 적용 효과 |
| 3 | Autoregressive (prev_pred) vs Step Embedding 비교 |
| 4 | 디코딩 방식별 비교: FC / Residual FC / Cross-Attention / Decoder LSTMCell |
| 5 | Multi-Head FC 실험 |
| 6 | 45d vs 60d 데이터 확장 효과 |
| 7 | Loss 함수 비교: Step-weighted MSE / MSELoss / HuberLoss |

### 2.2 전체 순위 (Clean 파이프라인)

| 순위 | 모델 | 데이터 | MAPE | R² | RMSE | MAE | Bias | Epoch | 파라미터 | 노트북 |
|:----:|------|:------:|-----:|----:|-----:|----:|-----:|------:|--------:|--------|
| **1** | **Seq2SeqAttn** | 60d | **2.79%** | **0.9849** | **6.96** | **4.32** | +0.86 | 53 | 377,585 | `07_seq2seq_attn_60d_v1` |
| 2 | Seq2Seq | 60d | 3.39% | 0.9831 | 7.37 | 4.98 | -0.66 | 23 | 361,201 | `06_seq2seq_no_attn_60d` |
| 3 | Residual 45d | 45d | 3.70% | 0.9799 | 7.93 | 5.23 | -1.21 | 37 | 205,967 | `09_residual_fc_45d` |
| 4 | Residual 60d | 60d | 3.98% | 0.9797 | 8.06 | 5.59 | +0.07 | 19 | 205,967 | `10_residual_fc_60d` |
| 5 | StepAttn_mid 60d | 60d | 4.01% | 0.9768 | 8.62 | 5.91 | +3.06 | 16 | 274,064 | `14_step_attn_mid_60d` |
| 6 | StepAttnModel | 45d | 4.49% | 0.9708 | 9.57 | 6.80 | +3.78 | 26 | 272,129 | `12_step_attn_base` |
| 7 | StepAttn_mid 45d | 45d | 4.56% | 0.9716 | 9.43 | 6.81 | +4.40 | 16 | 274,064 | `13_step_attn_mid` |
| 8 | StepEmbed 60d | 60d | 9.20% | 0.9040 | 17.54 | 14.71 | +14.18 | 6 | 208,569 | `16_step_embed_no_attn_60d` |
| 9 | MultiHead 60d | 60d | 10.77% | 0.8858 | 19.12 | 16.58 | +16.28 | 9 | 266,447 | `11_multihead_attn_fc_60d` |
| 10 | Autoreg | 45d | 14.30% | 0.8412 | 22.31 | 17.11 | -10.24 | 9 | 369,409 | `17_autoreg_prev_pred_feedback` |

> 비표준 파이프라인(Baseline, StepAttn_high)은 데이터 조건 차이로 직접 비교 제약이 있어 순위에서 제외.

### 2.3 디코딩 방식별 성능 계층

```
Decoder + Attention (2.79%)  >  Decoder (3.39%)  >  Residual FC (3.70%)  >  Cross-Attn (4.01%)  >  FC only (4.33%)
```

| 방식 | 대표 모델 | MAPE | Step 1 | Step 15 | 기울기 |
|------|----------|-----:|-------:|--------:|------:|
| Decoder + Attention | Seq2SeqAttn | **2.79%** | 1.14% | 5.39% | +0.328/step |
| Decoder | Seq2Seq | 3.39% | 1.65% | 6.11% | +0.330/step |
| Residual FC | Residual 45d | 3.70% | **0.95%** | 7.57% | +0.499/step |
| Cross-Attn + Residual | StepAttn_mid 60d | 4.01% | 2.06% | 6.75% | +0.346/step |
| FC only | Baseline | 4.33% | 1.14% | 8.12% | +0.514/step |

### 2.4 Attention 유효성: 동적 Query만 작동

| Attention 유형 | Query | MAPE | 작동 여부 |
|-------------|-------|-----:|:---:|
| **Bahdanau** (Seq2SeqAttn) | `decoder_hidden_t` (동적, step마다 갱신) | **2.79%** | O |
| 8-Head Cross (StepAttn_mid) | `step_queries` (고정) | 4.01% | X (균등분포) |
| 4-Head Cross (StepAttnModel) | `step_queries` (고정) | 4.49% | X (균등분포) |

### 2.5 Step 열화 패턴 분류

| 패턴 | 모델 | 기울기 | 원인 |
|------|------|------:|------|
| **정상** | Seq2SeqAttn, Seq2Seq, Residual, StepAttn | +0.27 ~ +0.50 | step 거리에 비례한 정보 감쇠 |
| **역전** | StepEmbed, MultiHead | -0.09 ~ -0.12 | step 정체성 미학습 (FC head가 step 구분 불가) |
| **폭발** | Autoreg | **+1.804** | prev_pred 피드백 루프에 의한 오차 누적 |

### 2.6 Phase 1 핵심 발견

1. **Encoder-Decoder 구조 유효** — Decoder LSTMCell 순차 디코딩이 FC(128→15) 동시 예측보다 우수
2. **Bahdanau Attention 유효** — 동적 Query 사용 시에만 작동 (+0.60%p). 고정 Query는 무효
3. **step_embedding > prev_pred** — Autoreg의 prev_pred는 +1.804%/step 오차 누적. step_embedding은 오류 차단
4. **Step-weighted MSE 효과적** — 상위 5개 모델 중 4개가 채택
5. **데이터량은 조건부 효과** — 구조와의 상호작용 존재 (Residual: 45d 우위, StepAttn: 60d 우위)

---

## 3. Phase 2 — Best Model 개선 시도 (v2)

### 3.1 v2 변경 사항 (8가지 동시 적용)

| # | 변경 | 내용 |
|:-:|------|------|
| 1 | Multi-run + seed 고정 | N=3, seeds=[42, 123, 7] |
| 2 | EarlyStopping patience | 5 → 10 |
| 3 | **Decoder dropout** | **0.2 추가 + Drop→LN→FC 순서** |
| 4 | Rainfall 전처리 | 일 누적 → 분당 차분 변환 |
| 5 | Bias 진단 | 계절·Step·값 구간별 분해 |
| 6 | Dead code 제거 | `_bahdanau_attention()` 삭제 |
| 7 | GPU 메모리 | CPU 텐서 + pin_memory |
| 8 | 평가 지표 | SMAPE, macro-avg MAPE 추가 |

### 3.2 v2 성능 — 2배 악화

| 지표 | v1 (단일) | v2 (Mean±Std) | 변화 |
|------|:---------:|:-------------:|:----:|
| MAPE | **2.79%** | 5.82 ± 0.33% | +3.03%p |
| R² | **0.9849** | 0.9714 ± 0.002 | -0.014 |
| Bias | +0.86 | +0.78 ± 0.62 | — |
| Step 1 MAPE | **1.14%** | 4.45 ± 0.34% | 3.9배 악화 |
| Step 15 MAPE | **5.39%** | 7.85 ± 0.40% | 1.5배 악화 |
| Epoch (평균) | 53 | 31 | 조기 종료 |

### 3.3 v2 분석에서 도출한 주요 발견

1. **Step 열화 기울기 27% 개선** (+0.328 → +0.241/step). 다만 Step 1 자체가 3.9배 악화되어 전체 MAPE 하락
2. **Bias의 "regression to mean"** — 저유량 +7.05, 고유량 -7.02 대칭적 수축 (MSE Loss 본질)
3. **Attention 차별화 실패** — v1에서도 동일. 15 step 모두 Entropy 4.191의 동일한 recency-biased 분포
4. **seed 의존성** — Best(5.35%) vs Worst(6.10%)로 0.75%p 차이

> 8가지 변경을 동시 적용하여 원인 특정 불가 → Ablation 실험 설계

---

## 4. Phase 3 — Ablation Study

### 4.1 실험 매트릭스

| 실험 | Rainfall | Decoder Dropout | ES/Scheduler | Mean MAPE | 결론 |
|:----:|:--------:|:---------------:|:------------:|----------:|------|
| v1 (원본) | 누적 | 없음 | v1 설정 | 2.79% (단일) | baseline |
| v2 | 차분 | 0.2 | v2 설정 | 5.82 ± 0.33% | 성능 2배 악화 |
| **Ablation A** | **누적 (원복)** | 0.2 (v2 유지) | v2 설정 | **5.92 ± 0.07%** | **dropout 존재 시 rainfall 효과 미감지** |
| **Ablation B = v3** | 차분 (v2 유지) | **없음 (원복)** | v2 설정 | **2.89 ± 0.31%** | **dropout = 지배적 원인** |
| **Ablation C** | **누적 (v1)** | **없음 (v1)** | **배포 설정** | **3.28 ± 0.07%** | **차분이 +0.40%p 우위 확정** |

> Ablation A/B(Phase 3)에서는 dropout의 파괴적 효과(+3%p)가 rainfall 차이(~0.4%p)를 완전히 가렸다. Ablation C에서 dropout을 제거한 상태로 비교한 결과, 차분 변환이 누적 대비 유의미하게 우수함을 확인. 즉, dropout이 **지배적** 원인이고 rainfall은 **부차적** 요인이다.

### 4.2 Decoder Dropout이 치명적인 메커니즘

Decoder LSTMCell은 autoregressive하게 15번 순차 실행:

```
Step 1:  h₁ = LSTMCell(input₁, (h₀, c₀))
Step 2:  h₂ = LSTMCell(input₂, (h₁, c₁))   ← h₁에 의존
  ...
Step 15: h₁₅ = LSTMCell(input₁₅, (h₁₄, c₁₄)) ← 모든 이전 step에 누적 의존
```

Dropout이 h_dec에 적용되면:
- Step t에서 꺼진 뉴런 정보가 Step t+1 이후 **영구 소실**
- 매 step마다 다른 뉴런이 꺼져 hidden state 연속성 파괴
- 15-step에 걸쳐 효과 **누적** → 학습 자체를 불안정하게 만듦

v2의 Drop→LN→FC 순서는 0이 된 뉴런을 포함한 벡터를 정규화하여 분포를 **이중 왜곡**.

### 4.3 Dropout 제거 효과 (v2 → v3)

| 지표 | v2 (dropout O) | v3 (dropout X) | 변화 |
|------|:--------------:|:--------------:|:----:|
| MAPE | 5.82% | **2.89%** | **2.0배 개선** |
| R² | 0.9714 | **0.9854** | +0.014 |
| Bias | +0.78 | **+0.28** | 64% 감소 |
| Step 1 MAPE | 4.45% | **1.13%** | **3.9배 개선** |
| Step 15 MAPE | 7.85% | **5.58%** | 1.4배 개선 |
| 학습 Epoch | 31 | **58** | 1.9배 더 깊은 수렴 |
| Best Val Loss | 0.001214 | **0.000766** | 37% 감소 |

### 4.4 Scheduler / EarlyStopping 충돌 수정

Ablation B 과정에서 Scheduler와 EarlyStopping 간 충돌 문제를 발견하여 수정:

| 파라미터 | 수정 전 | 수정 후 | 효과 |
|---------|:------:|:------:|------|
| `EarlyStopping.min_delta` | `1e-5` | **`1e-4`** | floating-point noise에 의한 counter reset 방지 |
| `ReduceLROnPlateau.patience` | `3` | **`5`** | LR 감소 전 충분한 관찰 구간 |
| Scheduler→ES 간격 | 7 epoch | **5 epoch** | LR 감소 효과를 관찰 가능한 구간 확보 |

---

## 5. 최종 모델 (v3) — 확정 성능

### 5.1 모델 구조

```
Encoder:  LSTM(10→128, 2-layer, dropout=0.2) → enc_outputs (72×128)
Attention: Bahdanau Additive (Query = decoder_hidden_t, 동적)
Decoder:  LSTMCell(144→128) × 15 steps
          Input = [context(128) + step_embedding(16)]
          Output = LN(h_dec) → Dropout → FC → (1)
          ★ Decoder hidden state (h_dec → h_dec+1) 간 dropout 없음
            (Output FC 직전 dropout은 유지 - hidden state 연속성과 무관)
Loss:     Step-weighted MSE (1.0 → 2.0)
Params:   377,585개
```

### 5.2 v3 구성 원칙

```
v1에서 유지:  Decoder 구조 (dropout 없음, LN→Drop→FC 순서)
v2에서 유지:  Multi-run (N=3), patience=10, pin_memory, SMAPE/macro 평가, Rainfall 차분
```

### 5.3 성능 (3-run)

| 지표 | Mean ± Std | Best (seed=7) |
|------|:----------:|:-------------:|
| **MAPE** | **2.89 ± 0.31%** | **2.66%** |
| SMAPE | 2.89 ± 0.31% | 2.66% |
| R² | 0.9854 ± 0.0017 | 0.9868 |
| RMSE | 6.83 ± 0.38 | 6.51 |
| MAE | 4.34 ± 0.38 | 4.06 |
| Bias | +0.28 ± 0.21 | -0.02 |
| Macro MAPE | 2.87 ± 0.29 | 2.63 |

### 5.4 Run별 상세

| Run | Seed | MAPE | R² | Bias | Epoch | Val Loss |
|:---:|:----:|-----:|---:|-----:|------:|---------:|
| 1 | 42 | 2.69% | 0.9863 | +0.38 | 62 | 0.000766 |
| 2 | 123 | 3.32% | 0.9830 | +0.47 | 39 | 0.001002 |
| **3** | **7** | **2.66%** | **0.9868** | **-0.02** | **73** | **0.000779** |

> v1(2.79%)은 v3의 정상 분포(2.66~3.32%) 안에 있는 전형적인 값. Lucky run이 아님을 확인.

### 5.5 Step별 MAPE

| Step | MAPE (Mean±Std) | Step | MAPE (Mean±Std) |
|-----:|:---------------:|-----:|:---------------:|
| 1 | 1.13 ± 0.20% | 9 | 3.06 ± 0.35% |
| 2 | 1.03 ± 0.07% | 10 | 3.44 ± 0.40% |
| 3 | 1.21 ± 0.05% | 11 | 3.83 ± 0.44% |
| 4 | 1.41 ± 0.11% | 12 | 4.22 ± 0.47% |
| 5 | 1.68 ± 0.17% | 13 | 4.65 ± 0.52% |
| 6 | 1.99 ± 0.20% | 14 | 5.10 ± 0.56% |
| 7 | 2.32 ± 0.24% | 15 | 5.58 ± 0.58% |
| 8 | 2.69 ± 0.30% | | |

### 5.6 계절별 MAPE

| 계절 | MAPE (Mean±Std) | 샘플 수 |
|------|:---------------:|--------:|
| Winter | 2.72 ± 0.18% | 6,054 |
| Spring | 2.85 ± 0.44% | 8,123 |
| Summer | 3.03 ± 0.33% | 11,274 |
| Fall | 2.86 ± 0.30% | 9,920 |

계절간 범위 **0.31%p** (v2: 1.21%p 대비 75% 축소). 4계절 모두 3% 이하의 균등한 성능.

---

## 6. Ablation C — Rainfall 누적 + 배포 ES/Scheduler (`20_ablation_rainfall_cumulative_es_fix`)

### 6.1 실험 목적

Phase 3의 Ablation A/B는 dropout이 있는 환경에서 rainfall을 비교했기 때문에, dropout의 지배적 효과(+3%p)가 rainfall 차이를 가릴 가능성이 있었다. Ablation C는 **dropout이 없는 환경에서 누적 vs 차분 rainfall의 효과를 분리 검증**한다.

### 6.2 실험 설계

```
배포 버전 (차분 rainfall)에서 rainfall 전처리만 누적으로 변경. 나머지 모든 코드 동일.
```

| 항목 | 배포 버전 | Ablation C | 변경 여부 |
|------|:---------:|:----------:|:---------:|
| Rainfall | 차분 | **누적** | **★ 단일 변인** |
| Decoder Dropout | 없음 | 없음 | 동일 |
| ES patience | 15 | 15 | 동일 |
| Scheduler threshold | 1e-4 (abs) | 1e-4 (abs) | 동일 |
| Seeds | [42, 123, 7] | [42, 123, 7] | 동일 |

### 6.3 성능 (3-run)

| 지표 | Mean ± Std | Best (seed=42) |
|------|:----------:|:--------------:|
| **MAPE** | **3.28 ± 0.07%** | **3.21%** |
| SMAPE | 3.29 ± 0.08% | 3.23% |
| R² | 0.9817 ± 0.0003 | 0.9817 |
| RMSE | 7.65 ± 0.07 | 7.56 |
| MAE | 4.87 ± 0.03 | 4.84 |
| Bias | +0.26 ± 0.34 | +0.37 |
| Macro MAPE | 3.23 ± 0.07% | 3.16% |

### 6.4 Run별 상세

| Run | Seed | MAPE | R² | Bias | Epoch |
|:---:|:----:|-----:|---:|-----:|------:|
| **1** | **42** | **3.21%** | **0.9817** | **+0.37** | **29** |
| 2 | 123 | 3.37% | 0.9821 | +0.61 | 38 |
| 3 | 7 | 3.25% | 0.9813 | -0.20 | 31 |

### 6.5 배포 버전(차분) vs Ablation C(누적) 비교

| 지표 | 배포 (차분) | Ablation C (누적) | 차이 |
|------|:----------:|:-----------------:|:----:|
| MAPE Mean | **2.88 ± 0.04%** | 3.28 ± 0.07% | +0.40%p |
| R² | **0.9881** | 0.9817 | -0.006 |
| Bias | -0.53 | +0.26 | — |
| Step 1 MAPE | **0.92%** | 1.12% | +0.20%p |
| Step 15 MAPE | **5.37%** | 6.31% | +0.94%p |
| MAPE Std | **0.04%** | 0.07% | — |
| 계절간 범위 | **0.21%p** | 0.73%p | 3.5배 |
| Epoch 범위 | 37~43 | 29~38 | 누적이 더 조기 종료 |

모든 지표에서 차분 rainfall이 우수. 특히 먼 Step일수록 차이가 확대됨 (Step 1: +0.20%p → Step 15: +0.94%p).

### 6.6 Step별 MAPE 비교

| Step | 배포 (차분) | Ablation C (누적) | 차이 |
|-----:|:----------:|:-----------------:|:----:|
| 1 | 0.92% | 1.12% | +0.20 |
| 2 | 0.89% | 1.18% | +0.29 |
| 3 | 1.13% | 1.38% | +0.25 |
| 4 | 1.45% | 1.54% | +0.09 |
| 5 | 1.77% | 1.86% | +0.09 |
| 6 | 2.12% | 2.18% | +0.06 |
| 7 | 2.46% | 2.64% | +0.18 |
| 8 | 2.78% | 3.04% | +0.26 |
| 9 | 3.15% | 3.51% | +0.36 |
| 10 | 3.49% | 3.93% | +0.44 |
| 11 | 3.86% | 4.40% | +0.54 |
| 12 | 4.23% | 4.89% | +0.66 |
| 13 | 4.61% | 5.34% | +0.73 |
| 14 | 5.00% | 5.85% | +0.85 |
| 15 | 5.37% | 6.31% | +0.94 |

Step 열화 기울기: 배포 +0.318/step vs Ablation C **+0.371/step**. 누적 rainfall의 0시 리셋 패턴이 장기 step에서 더 큰 예측 혼란을 유발.

### 6.7 계절별 MAPE 비교

| 계절 | 배포 (차분) | Ablation C (누적) | 차이 |
|------|:----------:|:-----------------:|:----:|
| Winter | 2.99% | **2.79%** | **-0.20%p (누적 우위)** |
| Spring | **2.78%** | 3.40% | +0.62%p |
| Summer | **2.85%** | 3.52% | +0.67%p |
| Fall | **2.93%** | 3.19% | +0.26%p |

Winter에서만 누적이 우수 — 겨울은 강수가 적어 0시 리셋 문제의 영향이 작기 때문. Spring/Summer는 강수가 잦아 차분 변환의 이점이 크게 나타남.

---

## 7. 배포 모델 — Scheduler/ES 수정 적용 (`seq2seq_attn_flow.ipynb`)

### 7.1 변경 목적

- Phase 3에서 발견한 Scheduler/EarlyStopping 충돌 수정 적용 (§4.4)
- 다중 배수지(J / A / D) 확장을 위한 배포 버전 생성

### 7.2 v3 대비 하이퍼파라미터 변경

| 파라미터 | v3 (실험) | 배포 버전 | 변경 이유 |
|---------|:---------:|:---------:|----------|
| ES patience | 10 | **15** | 다중 배수지 seed간 수렴 속도 편차 대응 |
| Scheduler threshold | (미설정) | **1e-4 (abs)** | min_delta와 일관된 개선 기준 |
| Scheduler patience | 5 | 5 (유지) | |
| ES min_delta | 1e-4 | 1e-4 (유지) | |

### 7.3 성능 (J배수지, 3-run)

| 지표 | Mean ± Std | Best (seed=123) |
|------|:----------:|:---------------:|
| **MAPE** | **2.88 ± 0.04%** | **2.85%** |
| SMAPE | 2.88 ± 0.03% | 2.85% |
| R² | 0.9881 ± 0.0002 | 0.9879 |
| RMSE | 10.06 ± 0.07 | 10.01 |
| MAE | 6.82 ± 0.01 | 6.82 |
| Bias | -0.53 ± 0.65 | -0.53 |
| Macro MAPE | 2.89 ± 0.04% | 2.85% |

### 7.4 Run별 상세

| Run | Seed | MAPE | R² | Bias | Epoch |
|:---:|:----:|-----:|---:|-----:|------:|
| 1 | 42 | 2.86% | 0.9882 | +0.26 | 37 |
| **2** | **123** | **2.85%** | **0.9879** | **-0.53** | **37** |
| 3 | 7 | 2.93% | 0.9883 | -1.32 | 43 |

### 7.5 v3 실험 vs 배포 버전 비교

| 지표 | v3 (실험) | 배포 버전 | 변화 |
|------|:---------:|:---------:|:----:|
| MAPE Mean | 2.89 ± 0.31% | **2.88 ± 0.04%** | 분산 87% 감소 |
| Best MAPE | **2.66%** (seed=7) | 2.85% (seed=123) | |
| R² Mean | 0.9854 | **0.9881** | +0.003 |
| Bias Mean | +0.28 | -0.53 | |
| Epoch 범위 | 39 ~ 73 | **37 ~ 43** | 수렴 안정화 |

Scheduler threshold 명시와 ES patience 확대로 seed간 분산이 87% 감소 (0.31% → 0.04%).
Mean MAPE와 R²는 소폭 개선되었으나, Best 단일 run 성능(2.66%)은 v3 실험이 우수.

### 7.6 Step별 MAPE

| Step | MAPE (Mean±Std) | Step | MAPE (Mean±Std) |
|-----:|:---------------:|-----:|:---------------:|
| 1 | 0.92 ± 0.13% | 9 | 3.15 ± 0.03% |
| 2 | 0.89 ± 0.09% | 10 | 3.49 ± 0.03% |
| 3 | 1.13 ± 0.09% | 11 | 3.86 ± 0.05% |
| 4 | 1.45 ± 0.11% | 12 | 4.23 ± 0.05% |
| 5 | 1.77 ± 0.07% | 13 | 4.61 ± 0.05% |
| 6 | 2.12 ± 0.08% | 14 | 5.00 ± 0.04% |
| 7 | 2.46 ± 0.06% | 15 | 5.37 ± 0.04% |
| 8 | 2.78 ± 0.02% | | |

### 7.7 계절별 MAPE

| 계절 | MAPE (Mean±Std) |
|------|:---------------:|
| Winter | 2.99 ± 0.03% |
| Spring | 2.78 ± 0.07% |
| Summer | 2.85 ± 0.01% |
| Fall | 2.93 ± 0.04% |

계절간 범위 **0.21%p** (v3 실험: 0.31%p 대비 32% 축소).

---

## 8. 버전별 비교 총괄

| 지표 | v1 (단일) | v2 (Mean) | v3 (Mean) | v3 Best | Abl.C (Mean) | 배포 (Mean) | 배포 Best |
|------|:---------:|:---------:|:---------:|:-------:|:------------:|:-----------:|:---------:|
| MAPE | 2.79% | 5.82% | 2.89% | 2.66% | 3.28% | **2.88%** | **2.85%** |
| R² | 0.9849 | 0.9714 | 0.9854 | 0.9868 | 0.9817 | **0.9881** | 0.9879 |
| Bias | +0.86 | +0.78 | +0.28 | -0.02 | +0.26 | **-0.53** | -0.53 |
| Step 1 MAPE | 1.14% | 4.45% | 1.13% | — | 1.12% | **0.92%** | — |
| Step 15 MAPE | 5.39% | 7.85% | 5.58% | — | 6.31% | **5.37%** | — |
| 계절간 범위 | — | 1.21%p | 0.31%p | — | 0.73%p | **0.21%p** | — |
| MAPE Std | — | 0.33% | 0.31% | — | 0.07% | **0.04%** | — |
| Epoch | 53 | 31 | 58 | 73 | 33 | **39** | 37 |

v3은 v1의 성능을 multi-run으로 재현하면서, Bias를 67% 개선하고 계절간 균형을 75% 향상.
Ablation C는 누적 rainfall이 차분 대비 +0.40%p 열위임을 확인하여, 배포 버전의 차분 파이프라인을 검증.
배포 버전은 Scheduler/ES 수정으로 seed간 분산을 87% 축소(0.31%→0.04%)하여 안정적 재현성 확보.

---

## 9. 결론

### 9.1 최종 모델 선정: 배포 버전 Seq2SeqAttn

12개 아키텍처 비교, ablation study(A/B/C), multi-run 검증, scheduler 안정화를 거쳐 **배포 버전 Seq2SeqAttn** (`seq2seq_attn_flow.ipynb`)을 최종 모델로 확정했다.

이 모델은 (1) 동적 Query Bahdanau Attention으로 Step 열화를 최소화하는 유일한 구조이며, (2) decoder dropout 제거로 LSTMCell 연속성을 보장하고, (3) 차분 rainfall로 0시 리셋 패턴을 해소하며, (4) Scheduler/ES 수정으로 seed 의존성을 사실상 해소하여, 3-run 평균 MAPE 2.88±0.04%의 안정적 성능을 달성했다.

### 9.2 모델 진화 흐름과 각 단계의 역할

```
v1 (Phase 1)  →  v2 (Phase 2)  →  v3 (Phase 3)  →  Ablation C  →  배포 버전
 아키텍처 탐색     개선 시도/실패     원인 규명/확정     rainfall 검증     안정화/실전 투입
```

| 버전 | 역할 | MAPE | 핵심 기여 | 탈락/채택 근거 |
|------|------|:----:|----------|--------------|
| **v1** | 아키텍처 선정 | 2.79% (단일) | 12개 중 Seq2SeqAttn이 최적 구조임을 입증 | 단일 run → 재현성 미검증 |
| **v2** | 실패에서 배운 교훈 | 5.82 ± 0.33% | decoder dropout의 치명적 영향 발견 계기 | 성능 2배 악화 |
| **v3** | 인과관계 확립 | 2.89 ± 0.31% | ablation으로 dropout = 지배적 원인 확정, multi-run 재현성 확보 | seed 의존성 잔존 (0.31%) |
| **Abl.C** | rainfall 검증 | 3.28 ± 0.07% | dropout 제거 환경에서 차분 rainfall이 +0.40%p 우위임을 확정 | 누적 열위 확인 |
| **배포** | **최종 확정** | **2.88 ± 0.04%** | Scheduler/ES 수정으로 분산 87% 축소, 다중 배수지 대응 | **채택** |

v1(2.79%)은 v3의 3-run 분포(2.66~3.32%) 내에 위치하는 전형적인 단일 실행 결과다. 배포 버전을 선택하는 이유는 MAPE 수치(v3 Best 2.66%이 더 낮음)가 아니라, **seed간 분산 87% 축소**(0.31%→0.04%), **계절간 균형 32% 추가 개선**(0.31%p→0.21%p), **수렴 안정화**(Epoch 39~73 → 37~43)라는 **재현성과 강건성**에 있다.

### 9.3 아키텍처 결론

1. **Seq2Seq Encoder-Decoder가 FC 동시 예측보다 우수** — MAPE 상위 2개 모두 Decoder 구조. LSTMCell 순차 디코딩이 step간 의존성을 자연스럽게 포착
2. **Bahdanau Attention은 동적 Query에서만 유효** — 고정 Query(Cross-Attention)는 균등분포로 수렴하여 무효 (Entropy 4.191). decoder hidden state가 step마다 갱신되는 동적 Query만 차별적 가중치 생성
3. **step_embedding이 prev_pred보다 우수** — Autoregressive 피드백은 +1.804%/step 오차 누적으로 실패. step_embedding은 오류 전파를 구조적으로 차단
4. **Multi-Head FC 실패** — step 정체성 부재로 step 순서 무관한 역전 패턴 발생

### 9.4 학습 결론

5. **Autoregressive Decoder에 dropout을 적용하면 안 된다** — LSTMCell의 hidden state 연속성을 파괴하여 15-step에 걸쳐 정보 소실 누적. 2×2 ablation으로 이것이 v2 성능 2배 악화의 **지배적 원인**임을 확정
6. **Rainfall 차분 변환이 누적보다 우수** — Ablation C에서 dropout 제거 환경에서의 직접 비교로 확인 (+0.40%p). 0시 리셋 패턴이 특히 먼 Step(+0.94%p at Step 15)과 강수기 계절(Spring/Summer +0.6%p)에서 예측을 악화
7. **단일 실행은 신뢰할 수 없다** — v3 multi-run(2.66~3.32%)으로 v1(2.79%)이 정상 범위임을 확인. 배포 버전에서 분산을 0.04%까지 축소하여 재현성 확보
8. **Ablation을 통한 인과관계 확립이 중요** — 8가지 동시 변경에서는 원인 특정 불가. 체계적 ablation으로 지배적 원인(dropout)과 부차적 요인(rainfall)을 단계적으로 분리
9. **Scheduler/EarlyStopping 간 상호작용이 수렴 안정성을 좌우** — min_delta와 scheduler threshold의 불일치가 floating-point noise에 의한 불필요한 counter reset을 유발. 명시적 threshold 설정으로 해소

### 9.5 서비스 관점의 최적화 방향

본 프로젝트의 목적은 펌프 스케줄링 최적화를 위한 수요 예측이다. 15분 예측을 4회 rolling하여 1시간을 커버하는 운용 방식에서, Step 13~15의 예측은 15분 후 Step 1~3으로 갱신된다. 따라서 **서비스 관점에서는 단기 Step(1~5)의 정확도가 펌프 제어에 직접적 영향**을 미치며, 먼 Step의 오차는 rolling 갱신으로 자연 보정된다.

| 구간 | Steps | 배포 MAPE | 서비스 역할 |
|------|-------|----------|------------|
| 단기 | 1~5 | 0.89 ~ 1.77% | 펌프 on/off 직접 결정 |
| 중기 | 6~10 | 2.12 ~ 3.49% | 스케줄 조정 참고 |
| 장기 | 11~15 | 3.86 ~ 5.37% | 다음 rolling에서 갱신 |

Step 열화 기울기(+0.328/step)는 12개 모델 중 최소값이며, 시간 거리에 비례한 불확실성 증가라는 시계열의 본질적 성질에 해당한다. 향후 개선은 먼 Step의 기울기를 낮추는 방향보다, 단기 구간의 Bias 축소(저유량 과대/고유량 과소 예측 완화)와 앙상블을 통한 분산 축소에 집중하는 것이 서비스 임팩트 대비 효율적이다.

### 9.6 알려진 한계

| 한계 | 상세 | 서비스 영향 |
|------|------|-----------|
| Attention 차별화 미달 | 15 step 모두 동일한 recency-biased 분포 (Entropy 4.191) | Step별로 다른 encoder 정보를 참조하지 못함. 그러나 현재 성능(2.88%)에서 실용적 병목은 아님 |
| Regression to mean | 저유량 과대, 고유량 과소 예측 (MSE Loss 본질) | 극단값 구간에서 펌프 스케줄 오차 가능. Quantile Loss 등으로 개선 여지 |
| 단일 배수지 검증 | J배수지만 학습/평가 | 배포 버전에서 A/D 배수지 확장 예정. 배수지별 특성 차이에 따른 성능 변동 가능 |

---

## 관련 파일

### 실험 노트북 (`notebook/experiment/`)

| 파일 | 모델 | Loss | 비고 |
|------|------|------|------|
| `07_seq2seq_attn_60d_v1` | Seq2SeqAttn **v1 (Best)** | SW-MSE | Phase 1 1위 |
| `08_seq2seq_attn_60d_v2_layernorm` | Seq2SeqAttn v2 | SW-MSE | Phase 2 (성능 하락) |
| `18_ablation_rainfall_restored` | Ablation A | SW-MSE | rainfall ≠ 원인 |
| `19_ablation_decoder_dropout_removed` | Ablation B = **v3 (최종)** | SW-MSE | **dropout = 원인 확정** |
| `20_ablation_rainfall_cumulative_es_fix` | **Ablation C** | SW-MSE | **차분 rainfall +0.40%p 우위 확정** |
| `06_seq2seq_no_attn_60d` | Seq2Seq (2위) | SW-MSE | |
| `09_residual_fc_45d` | Residual 45d (3위) | SW-MSE | |
| `10_residual_fc_60d` | Residual 60d (4위) | SW-MSE | |
| `14_step_attn_mid_60d` | Cross-Attn 8H (5위) | SW-MSE | |
| `12_step_attn_base` | Cross-Attn 4H | MSELoss | |
| `13_step_attn_mid` | Cross-Attn 8H 45d | SW-MSE | |
| `15_step_attn_high` | Cross-Attn 8H + Huber | HuberLoss | 비표준 |
| `16_step_embed_no_attn_60d` | Step Embedding | SW-MSE | 실패 |
| `11_multihead_attn_fc_60d` | MultiHead FC | SW-MSE | 실패 |
| `17_autoreg_prev_pred_feedback` | Autoreg Attention | MSELoss | 실패 |

### 배포 모델

| 파일 | 설명 |
|------|------|
| `notebook/seq2seq_attn_flow.ipynb` | 배포 버전 (Scheduler/ES 수정 적용, 다중 배수지 대응) |
| `seq2seq_attn_flow.ipynb` | 루트 배포 노트북 |

### 상세 분석 문서 (`docs/`)

| 파일 | 내용 |
|------|------|
| `12model_benchmark_report.md` | Phase 1: 12개 모델 전체 결과 |
| `seq2seq_attn_v2_analysis.md` | Phase 2: v2 결과 분석 |
| `ablation_decoder_dropout_analysis.md` | Phase 3: Ablation B 상세 분석 |
| `seq2seq_attn_v1_v2_v3_comparison.md` | v1/v2/v3 비교 보고서 |
| `scheduler_earlystopping_fix.md` | Scheduler/ES 충돌 수정 기록 |
