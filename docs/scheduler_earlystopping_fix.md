# Scheduler / EarlyStopping 충돌 수정 및 seq2seq_attn_flow_prediction 적용

## 1. 문제 — Scheduler와 EarlyStopping의 충돌(Collision)

### 기존 설정 (Ablation_B 초기 버전)

| 파라미터 | 값 |
|---------|-----|
| `EarlyStopping.min_delta` | `1e-5` |
| `ReduceLROnPlateau.patience` | `3` |
| `EarlyStopping.patience` | `10` |

**충돌 원인:**

- `min_delta=1e-5`는 개선으로 인정하는 최소 폭이 너무 작아, 미세한 수치 오차(floating-point noise) 수준의 변화도 "개선"으로 카운트됨
- Scheduler가 LR을 낮춘 직후 Loss가 소폭 감소(1e-5 이하)하면 ES counter가 reset되어, 실질적 개선 없이 학습이 계속 진행되는 현상 발생
- Scheduler patience=3 / ES patience=10 간격이 좁아, LR 감소 효과가 나타나기 전에 ES가 중복 발동될 위험 존재

---

## 2. 수정 내용

### 2.1 19_ablation_decoder_dropout_removed.ipynb

| 파라미터 | 변경 전 | 변경 후 | 효과 |
|---------|---------|---------|------|
| `EarlyStopping.min_delta` | `1e-5` | **`1e-4`** | 실질적 개선만 인정, noise에 의한 counter reset 방지 |
| `ReduceLROnPlateau.patience` | `3` | **`5`** | LR 감소 전 충분한 관찰 구간 확보 |
| `EarlyStopping.patience` | `10` | **`10`** (유지) | Scheduler(5) → ES(10) 간격 2배 확보 |

```python
# After
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
)
early_stopping = EarlyStopping(patience=10, min_delta=1e-4, verbose=False)
```

**간격 보장 논리:**
```
Scheduler patience=5  → 5 epoch 개선 없으면 LR × 0.5
ES patience=10        → 10 epoch 개선 없으면 학습 종료
간격 = 10 - 5 = 5 epoch (LR 감소 효과 관찰 가능 구간)
```

### 2.2 seq2seq_attn_flow_prediction.ipynb (신규 생성)

Ablation_B 구조를 기반으로 다중 배수지(J / A / D) 적용을 위해 생성.
Scheduler에 `threshold`와 `threshold_mode`를 명시하여 `min_delta`와 일관성을 유지.

| 파라미터 | 값 | 비고 |
|---------|-----|------|
| `ReduceLROnPlateau.patience` | `5` | Ablation_B와 동일 |
| `ReduceLROnPlateau.threshold` | `1e-4` | min_delta와 동일 기준 |
| `ReduceLROnPlateau.threshold_mode` | `'abs'` | 절댓값 기준 (상대값 아님) |
| `EarlyStopping.patience` | **`15`** | 더 긴 관찰 구간 확보 |
| `EarlyStopping.min_delta` | `1e-4` | Ablation_B와 동일 |

```python
# seq2seq_attn_flow_prediction.ipynb
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5,
    min_lr=1e-6, threshold=1e-4, threshold_mode='abs'   # ← 추가
)
early_stopping = EarlyStopping(patience=15, min_delta=1e-4, verbose=False)
```

**ES patience=15로 확대한 이유:**
다중 배수지 실험에서 seed에 따른 수렴 속도 편차가 크므로, 느린 seed도 충분히 학습할 수 있도록 여유 확보.

---

## 3. 설정 비교 요약

| | Ablation_B (수정 전) | Ablation_B (수정 후) | seq2seq_attn_flow_prediction |
|:---:|:---:|:---:|:---:|
| `min_delta` | `1e-5` | `1e-4` | `1e-4` |
| Scheduler patience | `3` | `5` | `5` |
| Scheduler threshold | — | — | `1e-4 (abs)` |
| ES patience | `10` | `10` | `15` |
| Scheduler→ES 간격 | `7` | **`5`** | **`10`** |

---

## 4. 검증 결과 (seq2seq_attn_flow_prediction.ipynb, D배수지 reservoir/33)

| Run | Seed | MAPE | SMAPE | R² | Bias | Epoch |
|:---:|:----:|-----:|------:|---:|-----:|------:|
| 1 | 42 | 3.15% | 3.15% | 0.9886 | +0.03 | 30 |
| 2 | 123 | 3.40% | 3.43% | 0.9875 | +0.02 | 25 |
| **3** | **7** | **3.06%** | **3.04%** | **0.9892** | **−0.12** | **36** |
| **Mean** | — | **3.20%** | **3.21%** | **0.9884** | **−0.02** | **30** |

수정된 설정에서 Bias가 −0.02로 거의 0에 수렴하여 과적합 없는 안정적 학습 확인.

---

## 5. 관련 파일

| 파일 | 설명 |
|------|------|
| [notebook/experiment/19_ablation_decoder_dropout_removed.ipynb](../notebook/experiment/19_ablation_decoder_dropout_removed.ipynb) | 수정 적용된 원본 |
| [notebook/seq2seq_attn_flow_prediction.ipynb](../notebook/seq2seq_attn_flow_prediction.ipynb) | 다중 배수지 확장 버전 |
| [docs/ablation_decoder_dropout_analysis.md](ablation_decoder_dropout_analysis.md) | Ablation B 전체 실험 분석 |
