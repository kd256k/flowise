# segment_id 버그 리포트 (2026-02-19)

## 1. 문제 요약

`segment_id`가 weather → merge → sliding window 파이프라인에서 전달되지 않아, **block 내 시간 gap(>1분)을 넘는 sliding window가 생성**되는 버그. LSTM이 시간적으로 불연속인 데이터를 연속 시계열로 학습하게 됨.

---

## 2. 발견된 문제 3가지

### 문제 1: `df_weather`에서 `segment_id` 누락

- Cell 14(`a7064ebf`)에서 weather 전처리 시 `segment_id` 생성
- Cell 18(`3b0f6d19`)에서 `df_weather` 복사 시 해당 컬럼 제외
- 이후 `df_merged`에도 `segment_id` 부재

```python
# Before (버그)
df_weather = df[['datetime', 'temperature', 'rainfall', 'humidity']].copy()

# After (수정)
df_weather = df[['datetime', 'temperature', 'rainfall', 'humidity', 'segment_id']].copy()
```

### 문제 2: merge 후 `segment_id` 미재계산

- weather의 `segment_id`는 weather 단독 기준
- flow+weather inner merge 후 flow 쪽 결측으로 추가 시간 gap 발생 가능
- merge 후 실제 시간 불연속을 반영하여 재계산 필요

```python
# Cell 21(6c0b5e38)에 추가
time_diff_merged = df_merged['time'].diff()
seg_boundary_merged = time_diff_merged > pd.Timedelta(minutes=1)
df_merged['segment_id'] = seg_boundary_merged.cumsum()
```

### 문제 3: sliding window에 `segment_id` 미적용

- Cell 28(`cboyx24gome`)에서 `block_id` 단일 루프로 연속 슬라이싱
- block 내 시간 gap이 있어도 무시하고 window 생성 → LSTM이 가짜 패턴 학습

```python
# Before: block_id 단일 루프
for block_id in sorted(df_merged['block_id'].unique()):
    block = df_merged[...].reset_index(drop=True)
    for i in range(n_samples):
        X_block.append(block_features[i : i + input_time])  # segment 경계 무시

# After: block_id × segment_id 이중 루프
for block_id in sorted(df_merged['block_id'].unique()):
    block = df_merged[...]
    for seg_id in sorted(block['segment_id'].unique()):
        segment = block[block['segment_id'] == seg_id].reset_index(drop=True)
        if len(segment) < min_segment_len:
            skipped_segments += 1
            continue
        for i in range(n_samples):
            seg_X.append(seg_features[i : i + input_time])  # segment 내에서만
```

---

## 3. 오염 영향도 정량 분석 (60d 데이터 기준)

### 3-1. 전체 규모

| 항목 | 수치 |
|------|------|
| 기존 총 윈도우 | 322,964개 |
| **오염된 윈도우 (segment 경계 초과)** | **82,542개 (25.56%)** |
| 수정 후 유효 윈도우 | 240,422개 |
| 스킵된 짧은 segment (< 87 steps) | 1,557개 |

### 3-2. 시즌별 오염률

| 시즌 | Segments | 오염 윈도우 | 오염률 |
|------|----------|------------|--------|
| **Winter** | 798 | 39,228 | **48.58%** |
| Spring | 1,048 | 23,404 | 29.74% |
| Fall | 231 | 13,055 | 16.25% |
| Summer | 83 | 6,855 | 8.24% |

### 3-3. Train/Val/Test 분할별 오염률

| 분할 | Winter | Spring | Summer | Fall |
|------|--------|--------|--------|------|
| **Train** | **53.72%** | 27.89% | 5.90% | 11.26% |
| **Val** | 20.55% | 29.49% | 16.94% | 24.69% |
| **Test** | 42.80% | **47.98%** | 12.35% | 31.81% |

### 3-4. 영향 해석

- Winter 학습 데이터의 과반(53.72%)이 오염 — 겨울철 학습 자체가 노이즈 위에서 진행
- Spring/Winter Test set 오염률 40%+ — 평가 지표(MAPE, R², RMSE)가 시간 불연속 window에 대한 예측을 포함
- 오염 window에서 LSTM은 72분 input 내 수십~수백분의 시간 점프를 연속 시계열로 처리
- **보고서(`12model_benchmark_report.md`)의 수치는 "깨끗한 시계열에 대한 정확한 평가"가 아님**

---

## 4. 수정 적용 현황

### 수정 완료 파일 (15개, 모두 동일한 3곳 수정)

| 그룹 | 파일 | Fix3 변형 |
|------|------|-----------|
| 60d (Seq2Seq) | `06_seq2seq_no_attn_60d.ipynb`, `07_seq2seq_attn_60d_v1.ipynb` | gap split |
| 60d (기존) | `10_residual_fc_60d.ipynb`, `11_multihead_attn_fc_60d.ipynb`, `16_step_embed_no_attn_60d.ipynb`, `14_step_attn_mid_60d.ipynb` | gap split |
| 45d | `09_residual_fc_45d.ipynb` | gap split |
| 이전 버전 | `12_step_attn_base.ipynb`, `13_step_attn_mid.ipynb`, `15_step_attn_high.ipynb`, `17_autoreg_prev_pred_feedback.ipynb`, `LSTM_flow_multi_AB.ipynb` | gap split |
| 이전 버전 (no-gap) | `LSTM_flow_multi.ipynb`, `seq2seq_attn_flow_prediction.ipynb`, `LSTM_weather_time_D.ipynb` | no-gap split |

### 미수정 파일 (segment_id 미사용 — weather 데이터 미포함)

| 파일 | 사유 |
|------|------|
| `01_lstm_flow_only_144to10.ipynb` | flow 단독, weather merge 없음 |
| `LSTM_flow_D.ipynb` | flow 단독, weather merge 없음 |
| `03_lstm_flow_131k_hidden32.ipynb` | flow 단독, weather merge 없음 |
| `02_lstm_flow_600samples.ipynb` | flow 단독, weather merge 없음 |

---

## 5. 재실험 필요 여부

### 영향받는 기존 실험 결과 (`12model_benchmark_report.md`)

| 모델 | 보고된 MAPE | 신뢰도 |
|------|------------|--------|
| Residual 60d (Best, 2차) | 3.38% | **낮음** — 오염된 window 기반 학습/평가 |
| Residual 60d (3차) | 3.54% | **낮음** |
| Residual 45d | 3.68% | **낮음** |
| Mid (Attn 8H) 60d | 3.88% | **낮음** |
| Multi-Head FC 60d | 9.50% | **낮음** — 실패 결론은 유지 가능 |

### 이미 수정된 상태로 학습/평가 완료

| 모델 | MAPE | 비고 |
|------|------|------|
| LSTMSeq2Seq 60d | 3.39% | segment-aware window로 학습 |
| LSTMSeq2SeqAttn 60d | 2.79% | segment-aware window로 학습 |

### 결론

- **기존 모델(Residual, StepAttn, MultiHead, StepEmbed)은 코드가 수정되었으나 재학습 전까지 기존 결과 유효하지 않음**
- Seq2Seq 계열 2개만 수정된 파이프라인으로 학습된 상태
- 기존 모델 간 상대 비교(예: Attention 유무)는 동일 조건 오염이므로 경향성은 참고 가능하지만, 절대 수치는 재측정 필요