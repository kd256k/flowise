# 1. 데이터 분석
- [X] 데이터 수집
    - [X] 업체 데이터 (유량 943,434건)
    - [X] 부가정보1 (기상 1,885,020건)
- [X] 데이터 전처리
    - [X] IQR 이상치 제거
    - [X] Linear Interpolation
    - [X] Savitzky-Golay 평활화
    - [X] Flow + Weather inner join (872,723건)
- [X] Feature Engineering
    - [X] 시간/요일/계절 cyclical encoding (sin/cos)
    - [X] 계절별 Block Sampling (60일 × 4계절)

# 2. 딥러닝 모델 실험
- [X] Phase 1: 12개 아키텍처 벤치마크
    - [X] Seq2Seq, Attention, Residual FC, Step Embedding 등 비교
    - [X] segment_id 버그 발견 → clean 파이프라인 재실험
    - [X] Best Model 선정: Seq2SeqAttn (MAPE 2.79%)
- [X] Phase 2: Best Model 개선 시도 (v2)
    - [X] 8가지 동시 변경 적용 → 성능 2배 악화 (5.82%)
    - [X] v2 분석 보고서 작성
- [X] Phase 3: Ablation Study (A/B)
    - [X] Ablation A: Rainfall 원복 → 원인 아님 확인
    - [X] Ablation B: Decoder dropout 제거 → 지배적 원인 확정
    - [X] v3 모델 확정 (MAPE 2.89±0.31%)
- [X] Phase 4: Ablation C + 최종 안정화
    - [X] 차분 rainfall +0.40%p 우위 확인
    - [X] Scheduler/EarlyStopping 안정화
    - [X] 최종 모델 확정 (MAPE 2.88±0.04%, 분산 87% 축소)

# 3. API 서빙
- [X] FastAPI 엔드포인트 구현 (flow-api, weather-api)
- [X] Redis 기반 비동기 태스크 관리
- [X] Docker / docker-compose 구성
- [X] 입력 유효성 검증 (4xx 반환)

# 4. 보고서
- [X] 12개 모델 벤치마크 보고서
- [X] v2 분석 보고서
- [X] Ablation Decoder Dropout 분석
- [X] Scheduler/ES 안정화 보고서
- [X] 최종 실험 보고서 (experiment_report.md)

# 5. GitHub 공개 준비
- [X] .gitignore 작성
- [X] README.md 작성
- [X] .env.example IP 마스킹
- [X] docker-compose.yml IP 마스킹
- [ ] LICENSE 파일 생성
- [ ] git init → 첫 커밋 → push
