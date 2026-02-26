# Flowise — 배수지 수요예측 AI 시스템

배수지 유출유량을 LSTM Seq2Seq + Bahdanau Attention으로 예측하여,
저비용 전기 시간대의 펌프 스케줄링을 최적화하는 시스템.

## 주요 성과

- **12개 아키텍처 비교** → Seq2SeqAttn 선정 (MAPE 2.79%)
- **Ablation Study** → Decoder dropout이 성능 2배 악화의 지배적 원인임을 확정
- **최종 모델** → 3-run 평균 MAPE **2.88 ± 0.04%**, 4계절 균등 성능

## 시스템 구성
```
┌──────────────┐     ┌───────────┐      ┌──────────────┐
│   flow-api   │────▶│   Redis   │◀────│ weather-api  │
│  (port 8000) │     │           │      │ (port 8001)  │
└──────────────┘     └─────┬─────┘      └──────────────┘
                     ┌─────┴─────┐
                     │   MySQL   │
                     └───────────┘
```

## 디렉토리 구조
```
├── src/                    # FastAPI 서빙 코드
│   ├── main.py             # FastAPI 엔트리포인트
│   ├── generator.py        # DB 조회 + 예측 파이프라인
│   ├── inference.py        # 모델 추론 서비스
│   ├── flowpredictor.py    # LSTM 모델 클래스
│   └── seq2seq_predictor.py # Seq2Seq+Attention 모델 클래스
├── notebook/               # 실험 노트북
│   ├── experiment/         # 12개 모델 실험 + Ablation A/B/C (20개)
│   └── seq2seq_attn_flow.ipynb  # 배포 버전 학습 노트북
├── data/                   # 원천 데이터 (git 제외)
├── models/                 # 학습된 모델 가중치 (git 제외)
├── docs/                   # 분석 보고서
│   └── experiment_report.md
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## 실험 요약

| Phase | 내용 | 결과 |
|-------|------|------|
| 1 | 12개 아키텍처 벤치마크 | Seq2SeqAttn 1위 (MAPE 2.79%) |
| 2 | Best Model 개선 시도 (v2) | 8가지 동시 변경 → 2배 악화 |
| 3 | Ablation Study (A/B) | Decoder dropout = 지배적 원인 |
| 4 | Ablation C | 차분 rainfall +0.40%p 우위 확정 |
| 배포 | Scheduler/ES 안정화 | MAPE 2.88±0.04%, 분산 87% 축소 |

상세 실험 결과는 [`docs/experiment_report.md`](docs/experiment_report.md) 참조.

## 모델 구조
```
Encoder:  LSTM(10→128, 2-layer, dropout=0.2)
Attention: Bahdanau Additive (동적 Query)
Decoder:  LSTMCell(144→128) × 15 steps
Loss:     Step-weighted MSE (1.0 → 2.0)
Params:   377,585개
```

- Input: 72분 (10 features: 유량, 기온, 강수, 습도, 시간/요일/계절 sin/cos)
- Output: 15분 예측 × 4 rolling = 1시간

## 실행 방법

### Docker (권장)
```bash
# .env 파일 생성
cp .env.example .env
# .env 내 DB/Redis 접속 정보 수정

# API 서비스 실행
docker compose up -d flow-api weather-api

# 개발 환경 (Jupyter 포함)
docker compose --profile dev up -d
```

### 로컬 실행
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

uvicorn src.main:app --host 0.0.0.0 --port 8000
```

## 기술 스택

- **ML**: PyTorch 2.x, CUDA 12.x
- **API**: FastAPI, Uvicorn
- **Infra**: Docker, Redis, MySQL
- **Data**: pandas, scikit-learn, SciPy (Savitzky-Golay)

## 라이센스

© 2024-2025. All Rights Reserved.

본 프로젝트는 포트폴리오 목적으로 공개되었으며, 무단 복제 및 상업적 사용을 금합니다.