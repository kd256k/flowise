# LSTM 유량 예측 — 실험 결과 보고서 (2025-02-19)

> **segment_id 버그 수정 완료 (2025-02-20)**
> 전체 12개 모델 clean 파이프라인(segment-aware sliding window) 재실험 완료. 모든 결과 확정.
> 상세: [segment_id_bug_report.md](segment_id_bug_report.md)

---

## 실험 목적

1. Encoder-Decoder (Seq2Seq) 구조의 유효성 검증
2. Bahdanau Attention의 Seq2Seq 적용 효과 확인
3. Autoregressive (prev_pred) vs Step Embedding 비교
4. 60일 데이터 확장 효과 확인 (45d → 60d)
5. Step 열화 분석 (먼 step의 예측 정확도 감소 정량화)
6. Multi-Head FC 구조 실험 (step별 독립 FC head)
7. 디코딩 방식별 비교: FC 동시 예측 / Residual FC / Cross-Attention / Decoder LSTMCell
8. Loss 함수 영향 비교: Step-weighted MSE / MSELoss / HuberLoss

---

## 공통 설정

| 항목 | 값 |
|------|-----|
| input_time | 72분 (1.2시간), 10 features |
| output_time | 15분 (× 4 rolling = 1시간) |
| Features | value, temperature, rainfall, humidity, time_sin/cos, dow_sin/cos, season_sin/cos |
| Data Split | Train 70% / Val 15% / Test 15% (block별 독립, gap=87) |
| 정규화 | Train 기준 MinMaxScaler (sin/cos 제외) |
| Early Stopping | patience=5 |
| Optimizer | Adam (lr=0.001, weight_decay=1e-5) |
| Gradient Clipping | max_norm=1.0 |

### 데이터 파이프라인

| 항목 | Buggy (수정 전) | Clean (segment-aware) |
|------|:---:|:---:|
| 60d 총 윈도우 | 322,268 | **239,726** |
| 45d 총 윈도우 | 240,900 | **180,886** |
| 오염 윈도우 비율 | 25.56% | 0% |
| 세그먼트 경계 처리 | 무시 | segment별 독립 생성 |

### 계절별 Block Sampling (60일 × 4, 표준)

| 계절 | 기간 |
|------|------|
| Winter | 2023-12-01 ~ 2024-01-29 |
| Spring | 2024-03-15 ~ 2024-05-13 |
| Summer | 2023-07-01 ~ 2023-08-29 |
| Fall | 2023-09-15 ~ 2023-11-13 |

> **비표준 파이프라인 주의**
> - **Baseline** (`seq2seq_attn_flow_prediction`): weather 22파일(표준 23), log1p rainfall, Spring 2023-04-01~05-15, split gap 없음, 175,591 windows
> - **StepAttn_high**: Fall 2023-09-01~10-30(표준: 09-15~11-13), HuberLoss(delta=0.01), 241,603 windows
> - 이 두 모델은 데이터 차이로 다른 모델과의 직접 비교에 제약 있음

---

## 1. 전체 Clean 결과 순위

| 순위 | 모델 | 데이터 | MAPE | R² | RMSE | MAE | Bias | Epoch | 파라미터 |
|:----:|------|:------:|-----:|----:|-----:|----:|-----:|------:|--------:|
| **1** | **Seq2SeqAttn 60d** | 60d | **2.79%** | **0.9849** | **6.96** | **4.32** | **+0.86** | 53 | 377,585 |
| 2 | Seq2Seq 60d | 60d | 3.39% | 0.9831 | 7.37 | 4.98 | -0.66 | 23 | 361,201 |
| 3 | Residual 45d | 45d | 3.70% | 0.9799 | 7.93 | 5.23 | -1.21 | 37 | 205,967 |
| 4 | Residual 60d | 60d | 3.98% | 0.9797 | 8.06 | 5.59 | +0.07 | 19 | 205,967 |
| 5 | StepAttn_mid 60d | 60d | 4.01% | 0.9768 | 8.62 | 5.91 | +3.06 | 16 | 274,064 |
| ※6 | Baseline | 비표준 | 4.33% | 0.9739 | 9.18 | 6.18 | -2.75 | 19 | 205,711 |
| 7 | StepAttnModel 45d | 45d | 4.49% | 0.9708 | 9.57 | 6.80 | +3.78 | 26 | 272,129 |
| 8 | StepAttn_mid 45d | 45d | 4.56% | 0.9716 | 9.43 | 6.81 | +4.40 | 16 | 274,064 |
| ※9 | StepAttn_high | 60d† | 7.26% | 0.9126 | 16.49 | 11.64 | +7.61 | 16 | 272,129 |
| 10 | StepEmbed 60d | 60d | 9.20% | 0.9040 | 17.54 | 14.71 | +14.18 | 6 | 208,569 |
| 11 | MultiHead 60d | 60d | 10.77% | 0.8858 | 19.12 | 16.58 | +16.28 | 9 | 266,447 |
| 12 | Autoreg 45d | 45d | 14.30% | 0.8412 | 22.31 | 17.11 | -10.24 | 9 | 369,409 |

> - ※6 Baseline: 비표준 파이프라인 — 참고 순위
> - ※9 StepAttn_high: 비표준 Fall 범위 + HuberLoss — 참고 순위
> - 12위 Autoreg: 학습 불안정으로 실행마다 결과 편차 큼 (이전 실행 MAPE 8.25%)

---

## 2. Buggy vs Clean 비교

오염 윈도우(25.56%) 제거 + 데이터량 감소(-25%) 후 성능 변화:

| 모델 | Buggy MAPE | Clean MAPE | Δ MAPE | 방향 |
|------|:---------:|:---------:|:------:|:----:|
| MultiHead 60d | 9.50% | 10.77% | +1.27%p | 악화 |
| StepAttn_mid 45d | 3.92% | 4.56% | +0.64%p | 악화 |
| Residual 60d | 3.38% | 3.98% | +0.60%p | 악화 |
| StepAttn_mid 60d | 3.88% | 4.01% | +0.13%p | 악화 |
| Residual 45d | 3.68% | 3.70% | +0.02%p | 유지 |
| StepAttn_high | 7.65% | 7.26% | -0.39%p | 개선 |
| StepAttnModel 45d | 4.89% | 4.49% | -0.40%p | 개선 |
| StepEmbed 60d | 12.15% | 9.20% | -2.95%p | 개선 |

> Seq2SeqAttn, Seq2Seq, MultiHead, Autoreg는 원래 clean 파이프라인으로 실험.
> Baseline은 buggy 기준 MAPE 미기록.

### 분석

- **악화 (4개)**: 데이터량 25% 감소가 주 원인. 학습이 충분히 수렴한 모델(Residual, StepAttn_mid)은 데이터 감소 영향이 더 큼
- **개선 (3개)**: 오염 윈도우가 학습을 적극적으로 방해하던 모델. StepEmbed(-2.95%p) 개선 폭이 가장 큼
- **순위 패턴 유지**: Buggy/Clean 상관없이 상위 모델 그룹(Residual, StepAttn_mid)의 상대적 위치 유지

---

## 3. Encoder-Decoder (Seq2Seq) 구조 검증

### 아키텍처 비교

```
Seq2Seq:      Encoder LSTM(2L, 128) → context_proj → Decoder LSTMCell × 15
              dec_input = [context, step_embedding(step_idx)]
              step 정체성: step_embedding (16-dim, 학습 가능)

Seq2SeqAttn:  Encoder LSTM(2L, 128) → Bahdanau Attention → Decoder LSTMCell × 15
              Query = decoder_hidden_t (동적, step마다 갱신)
              dec_input = [context_vector, step_embedding(step_idx)]

Autoreg:      Encoder LSTM(2L, 128) → Bahdanau Attention → Decoder LSTMCell × 15
              dec_input = [context_vector, prev_pred]
              prev_pred = 이전 step의 자기 예측값 (error feedback)
```

### 핵심 차이: step_embedding vs prev_pred

| 구분 | Seq2Seq / Seq2SeqAttn | Autoreg |
|------|:---:|:---:|
| Decoder 입력 | `step_embedding(idx)` | `prev_pred` (자기 예측) |
| Step 정보 전달 | 학습된 위치 임베딩 | 이전 예측값 |
| 오류 전파 | **차단** (각 step 독립) | **누적** (에러 피드백 루프) |
| Scheduled Sampling | 불필요 | 필요 (tf_ratio 1.0→0.0) |

### 결과 비교

| 지표 | Seq2SeqAttn | Seq2Seq | Autoreg |
|------|----------:|--------:|--------:|
| MAPE | **2.79%** | 3.39% | 14.30% |
| R² | **0.9849** | 0.9831 | 0.8412 |
| Step 1 MAPE | 1.14% | 1.65% | 4.65% |
| Step 15 MAPE | **5.39%** | 6.11% | 27.43% |
| Step 기울기 | +0.328%/step | +0.330%/step | **+1.804%/step** |
| Epoch | 53 | 23 | 9 (Early Stop) |

### 결론

1. **Seq2Seq Encoder-Decoder 구조 유효** — Decoder LSTMCell의 순차 디코딩이 FC(128→15) 동시 예측보다 우수
2. **Bahdanau Attention 유효** — Seq2Seq 대비 MAPE 0.60%p 추가 개선. 동적 Query가 step마다 encoder의 다른 시점에 집중
3. **step_embedding이 prev_pred보다 우수** — Autoreg의 prev_pred는 오류 누적(+1.804%/step)을 유발하여 Step 15 MAPE 27.43%로 실패
4. **Autoreg는 구조적 실패 + 학습 불안정** — 이전 실행(MAPE 8.25%)과 현재 실행(14.30%)의 편차가 6%p. 자기 회귀 구조의 높은 학습 variance 확인

---

## 4. Attention 메커니즘 유효성

### 기존 결론 (buggy 데이터): "Attention 무효"

- 8-Head Cross-Attention(StepAttn_mid)에서 weight가 균등(≈1/72) → Attention 미학습
- Residual(No Attn) 3.38% vs StepAttn_mid(8H) 3.88% → Attention 없는 모델이 더 우수

### 수정된 결론 (clean 데이터): "동적 Query Attention만 유효"

| Attention 유형 | 모델 | Query | MAPE | 작동 여부 |
|-------------|------|-------|-----:|:---:|
| **Bahdanau** (Seq2SeqAttn) | Decoder LSTMCell | `decoder_hidden_t` (동적) | **2.79%** | ✅ 유효 |
| 8-Head Cross (StepAttn_mid 60d) | FC head | `step_queries` (고정) | 4.01% | ❌ 균등분포 |
| 4-Head Cross (StepAttnModel) | FC head | `step_queries` (고정) | 4.49% | ❌ 균등분포 |

**핵심**: Attention 자체가 아니라 **Query가 동적으로 갱신되는지 여부**가 관건.
Bahdanau는 decoder 순차 실행 → 매 step의 hidden state가 Query → 차별화 성공.
Cross-Attention은 고정 step_queries → 학습 시 차별화 실패.

---

## 5. Multi-Head FC 실험 (실패)

### 구조

```
기존 (Residual):  LSTM(128) → LayerNorm → Dropout → FC(128→15)
Multi-Head:       LSTM(128) → LayerNorm → Dropout → [FC(128→32→ReLU→1)] × 15
                  FC 파라미터: 62,415개 (기존 ~2K의 31배)
```

### 결과

| 지표 | MultiHead (clean) |
|------|------------------:|
| MAPE | 10.77% |
| R² | 0.8858 |
| Step 1 MAPE | 14.35% (worst) |
| Step 15 MAPE | 11.27% |
| 기울기 | **-0.088%/step (역전)** |
| Epoch | 9 (Early Stop) |

### 실패 원인

1. **Step 정체성 부재** — 15개 head가 동일 hidden state를 받지만 "몇 번째 step인지" 정보 없음
2. **Step 간 상호작용 부재** — 독립 head로 인접 step 간 시간적 연속성 파괴
3. **Step-weighted Loss 역효과** — Step 15에 2배 가중치 → Step 1 학습 희생 (14.35%)

---

## 6. Step 열화 분석 (전 모델)

### Step 1 / Step 15 MAPE 및 기울기

| 순위 | 모델 | Step 1 | Step 15 | 기울기 | 비율 | Loss | 패턴 |
|:----:|------|-------:|--------:|------:|-----:|------|------|
| 1 | Seq2SeqAttn | 1.14% | 5.39% | +0.328 | 4.7x | SW-MSE | 정상 |
| 2 | Seq2Seq | 1.65% | 6.11% | +0.330 | 3.7x | SW-MSE | 정상 |
| 3 | Residual 45d | **0.95%** | 7.57% | +0.499 | **8.0x** | SW-MSE | 정상 (급격) |
| 4 | Residual 60d | 2.52% | 6.84% | +0.332 | 2.7x | SW-MSE | 정상 |
| 5 | StepAttn_mid 60d | 2.06% | 6.75% | +0.346 | 3.3x | SW-MSE | 정상 |
| ※6 | Baseline | **1.14%** | 8.12% | +0.514 | **7.1x** | MSELoss | 정상 (급격) |
| 7 | StepAttnModel | 3.73% | 7.06% | **+0.270** | **1.9x** | MSELoss | 정상 (완만) |
| 8 | StepAttn_mid 45d | 2.48% | 7.30% | +0.350 | 2.9x | SW-MSE | 정상 |
| ※9 | StepAttn_high | 4.90% | 9.97% | +0.378 | 2.0x | HuberLoss | 정상 |
| 10 | StepEmbed | 9.67% | 8.45% | -0.115 | 0.9x | SW-MSE | **역전** |
| 11 | MultiHead | 14.35% | 11.27% | -0.088 | 0.8x | SW-MSE | **역전** |
| 12 | Autoreg | 4.65% | 27.43% | **+1.804** | **5.9x** | MSELoss | **폭발** |

### 계절별 Step 열화 (Seq2SeqAttn, Best Model)

| 계절 | Step 1 | Step 15 | Mean MAPE | Bias |
|------|-------:|-------:|----------:|-----:|
| Winter | 1.02% | 5.75% | 2.85% | +2.67 |
| Spring | 1.18% | 5.02% | 2.73% | +0.82 |
| Summer | 1.16% | 5.59% | 3.08% | +0.68 |
| Fall | 1.15% | 5.25% | 2.70% | +0.11 |
| **All** | **1.14%** | **5.39%** | **2.79%** | **+0.86** |

### 핵심 패턴

1. **Step 1 최저 = LSTM hidden state 품질**: Residual 45d(0.95%), Baseline(1.14%), Seq2SeqAttn(1.14%). FC 직접 예측이 직후 1-step에서는 Decoder보다 유리
2. **기울기 최소 = 디코딩 메커니즘 품질**: StepAttnModel(+0.270) < Seq2SeqAttn(+0.328) ≈ Seq2Seq(+0.330) ≈ Residual 60d(+0.332). 단, StepAttnModel은 Step 1이 높아(3.73%) 전체 MAPE는 열위
3. **MAPE = Step 1 수준 × 기울기의 조합**: Seq2SeqAttn은 Step 1(1.14%)과 기울기(+0.328) 모두 우수하여 2.79% 달성
4. **역전 패턴 = 학습 실패**: StepEmbed, MultiHead에서 step 순서 무관한 불규칙 MAPE → step 정체성 미학습
5. **폭발 패턴 = 오차 누적**: Autoreg의 +1.804%/step은 prev_pred 피드백 루프에 의한 구조적 한계

---

## 7. 디코딩 방식별 분석

### FC 동시 예측 vs Decoder LSTMCell 순차 디코딩

| 방식 | 대표 모델 | MAPE | Step 1 | Step 15 | 기울기 | 비율 |
|------|----------|-----:|-------:|--------:|------:|-----:|
| **Decoder + Attention** | Seq2SeqAttn | **2.79%** | 1.14% | 5.39% | +0.328 | 4.7x |
| **Decoder** | Seq2Seq | 3.39% | 1.65% | 6.11% | +0.330 | 3.7x |
| **Residual FC** | Residual 45d | 3.70% | **0.95%** | 7.57% | +0.499 | 8.0x |
| **Residual FC** | Residual 60d | 3.98% | 2.52% | 6.84% | +0.332 | 2.7x |
| **Cross-Attn + Residual** | StepAttn_mid 60d | 4.01% | 2.06% | 6.75% | +0.346 | 3.3x |
| **FC only** | Baseline | 4.33% | 1.14% | 8.12% | +0.514 | 7.1x |

> **Decoder > Residual FC > Cross-Attn + Residual > FC only** (전체 MAPE 기준)

### 45d vs 60d 비교

| 모델 | 45d MAPE | 60d MAPE | Δ | 45d 기울기 | 60d 기울기 |
|------|:--------:|:--------:|:---:|:---------:|:---------:|
| Residual | **3.70%** | 3.98% | +0.28%p | +0.499 | +0.332 |
| StepAttn_mid | 4.56% | **4.01%** | -0.55%p | +0.350 | +0.346 |

- **Residual**: 45d가 MAPE 우위(3.70% vs 3.98%). 단 Step 1이 0.95%로 극도로 낮고 기울기(+0.499)가 가파름 → 근거리 예측에 편향
- **StepAttn_mid**: 60d가 MAPE 우위(4.01% vs 4.56%). 기울기는 거의 동일(~0.35)
- **데이터량 효과**: 단순히 더 많은 데이터 ≠ 더 좋은 성능. 모델 구조와의 상호작용 존재

### Loss 함수별 비교

| Loss | 모델 | MAPE | Step 15/1 비율 |
|------|------|-----:|:-----------:|
| Step-weighted MSE | Seq2SeqAttn | **2.79%** | 4.7x |
| Step-weighted MSE | Residual 60d | 3.98% | 2.7x |
| MSELoss (균등) | StepAttnModel | 4.49% | **1.9x** |
| MSELoss (균등) | Baseline | 4.33% | 7.1x |
| HuberLoss(δ=0.01) | StepAttn_high | 7.26% | 2.0x |

- **Step-weighted MSE**: 상위 모델 대부분 채택. 후반 step 가중으로 전체 MAPE 개선
- **MSELoss 균등 가중치**: 비율은 낮을 수 있으나(StepAttnModel 1.9x) Step 1 MAPE가 높아(3.73%) 전체 성능 열위
- **HuberLoss(δ=0.01)**: 정규화 [0,1] 범위에서 사실상 L1 Loss → gradient 약화로 학습 효율 저하

---

## 8. 결론 및 Best Model

### Best Model: LSTMSeq2SeqAttn (60d)

```
구조: Encoder LSTM(10→128, 2L, dropout=0.2)
      → Bahdanau Attention (Query=decoder_hidden_t)
      → Decoder LSTMCell × 15 (step_embedding 16-dim)
Loss: Step-weighted MSE (1.0~2.0)
LR Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)
파라미터: 377,585개
저장: models/lstm_seq2seq_attn_60d.pth
```

| 지표 | 값 |
|------|-----|
| MAPE | **2.79%** |
| R² | **0.9849** |
| RMSE | 6.96 |
| MAE | 4.32 |
| Bias | +0.86 |
| Step 1 MAPE | 1.14% |
| Step 15 MAPE | 5.39% |
| Step 기울기 | +0.328%/step |
| 학습 Epoch | 53 |

### 핵심 발견

1. **Encoder-Decoder (Seq2Seq) 구조 유효** — Decoder LSTMCell의 순차 디코딩이 FC(128→15) 동시 예측보다 우수. MAPE 상위 2개 모두 Decoder 구조
2. **Bahdanau Attention 유효** — 동적 Query(decoder_hidden_t) 사용 시 유효. 고정 Query(step_queries)는 무효
3. **step_embedding > prev_pred** — Autoreg의 prev_pred는 +1.804%/step 오류 누적 유발. step_embedding은 오류 차단으로 +0.328%/step 유지
4. **FC(128→15) Baseline의 이중성** — Step 1 MAPE 1.14%(공동 최저)이나 Step 15에서 8.12%(최악급). 디코딩 보정 메커니즘의 필요성을 정량적으로 증명
5. **Multi-Head FC / StepEmbed 실패** — step 정체성 없는 독립 head 또는 underfitting은 역전 패턴(step 무관한 MAPE) 초래
6. **Step-weighted MSE 효과적** — 후반 step 가중치 배정이 전체 MAPE 개선에 기여. 상위 5개 모델 중 4개가 채택
7. **데이터량은 조건부 효과** — StepAttn_mid는 60d에서 개선(-0.55%p), Residual은 45d에서 더 우수(+0.28%p). 구조와의 상호작용 존재

### 노트북 파일

| 파일 | 모델 | 파이프라인 | Loss | 상태 |
|------|------|:---:|------|------|
| `07_seq2seq_attn_60d_v1.ipynb` | Seq2SeqAttn (Best) | ✅ Clean | SW-MSE | 확정 |
| `06_seq2seq_no_attn_60d.ipynb` | Seq2Seq | ✅ Clean | SW-MSE | 확정 |
| `09_residual_fc_45d.ipynb` | Residual 45d | ✅ Clean | SW-MSE | 확정 |
| `10_residual_fc_60d.ipynb` | Residual 60d | ✅ Clean | SW-MSE | 확정 |
| `14_step_attn_mid_60d.ipynb` | Cross-Attn 8H 60d | ✅ Clean | SW-MSE | 확정 |
| `seq2seq_attn_flow_prediction.ipynb` | Baseline | ※ 비표준 | MSELoss | 참고 |
| `12_step_attn_base.ipynb` | Cross-Attn 4H | ✅ Clean | MSELoss | 확정 |
| `13_step_attn_mid.ipynb` | Cross-Attn 8H 45d | ✅ Clean | SW-MSE | 확정 |
| `15_step_attn_high.ipynb` | Cross-Attn 8H + Huber | ※ 비표준 | HuberLoss | 참고 |
| `16_step_embed_no_attn_60d.ipynb` | Step Embedding | ✅ Clean | SW-MSE | 확정 (실패) |
| `11_multihead_attn_fc_60d.ipynb` | MultiHead FC | ✅ Clean | SW-MSE | 확정 (실패) |
| `17_autoreg_prev_pred_feedback.ipynb` | Autoreg Attention | ✅ Clean | MSELoss | 확정 (실패) |