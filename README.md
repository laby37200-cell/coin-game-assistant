# 동전게임 실시간 공략 가이드

MuMu Player에서 실행되는 **'콘텐츠페이 동전게임'**(수박게임의 변형판)을 실시간으로 분석하여 최적의 동전 낙하 위치를 오버레이로 표시하는 AI 기반 공략 프로그램입니다.

![Game Screen](docs/images/game_screen.png)
*콘텐츠페이 동전게임 초기 화면*

![Coin Sizes](docs/images/coin_sizes.png)
*동전 크기 비교*

## 주요 기능

- **🤖 AI 화면 분석**: Google Gemini 3.0 Flash Preview 모델을 사용하여 게임 화면의 동전 위치와 종류를 자동 인식
- **⚙️ 물리 시뮬레이션**: Pymunk 2D 물리 엔진으로 동전이 떨어졌을 때의 결과를 정확하게 예측
- **🎯 최적 위치 계산**: 수박게임 전문 공략 알고리즘을 적용하여 가장 높은 점수를 얻을 수 있는 위치 제시
- **📊 실시간 오버레이**: 게임 화면 위에 투명한 가이드 라인으로 최적 위치를 직관적으로 표시
- **🔄 자동 상태 감지**: 동전이 움직이는 중에는 대기하고, 정지 상태일 때만 분석하여 API 비용 절감

## 기술 스택

### 언어 및 프레임워크
- **Python 3.10+**: 메인 개발 언어
- **tkinter**: 오버레이 UI

### AI/Vision
- **Google Generative AI SDK**: Gemini 3.0 Flash Preview 모델 사용
- **Pillow**: 이미지 처리
- **OpenCV**: 화면 분석

### 물리 엔진
- **Pymunk**: 2D 물리 시뮬레이션 (Chipmunk 기반)

### 화면 캡처
- **mss**: 고속 스크린샷 캡처
- **pygetwindow**: 윈도우 탐지 및 위치 추적

## 설치 방법

### 1. 필수 요구사항

- **Python 3.10 이상**
- **MuMu Player** (또는 다른 Android 에뮬레이터)
- **Google Gemini API 키** ([Google AI Studio](https://makersuite.google.com/app/apikey)에서 발급)

### 2. 패키지 설치

```bash
# 프로젝트 디렉토리로 이동
cd coin_game_assistant

# 의존성 패키지 설치
pip install -r requirements.txt
```

### 3. API 키 설정

환경 변수로 Gemini API 키를 설정합니다:

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY="your-api-key-here"
```

**Windows (CMD):**
```cmd
set GEMINI_API_KEY=your-api-key-here
```

**Linux/Mac:**
```bash
export GEMINI_API_KEY="your-api-key-here"
```

또는 `config.py` 파일에서 직접 설정할 수도 있습니다.

## 사용 방법

### 1. 게임 실행

1. **MuMu Player**를 실행합니다.
2. **'콘텐츠페이 동전게임'** 또는 유사한 수박게임 변형판을 시작합니다.
3. 게임이 플레이 가능한 상태가 될 때까지 대기합니다.

### 2. 프로그램 실행

```bash
python main.py
```

### 3. 사용

- 프로그램이 자동으로 MuMu Player 창을 찾아 오버레이를 표시합니다.
- 게임이 **정지 상태**가 되면 자동으로 화면을 분석합니다.
- **녹색 수직선**이 최적의 낙하 위치를 표시합니다.
- 해당 위치에 동전을 떨어뜨리세요!

### 4. 종료

- **Ctrl+C**를 누르거나
- **오버레이 창**을 닫으면 프로그램이 종료됩니다.

## 프로젝트 구조

```
coin_game_assistant/
├── main.py                 # 메인 실행 파일
├── config.py              # 설정 파일
├── requirements.txt       # 의존성 패키지
├── README.md             # 이 파일
│
├── models/               # 데이터 모델
│   ├── __init__.py
│   └── coin.py          # 동전 계층 구조 정의
│
├── vision/              # 화면 분석
│   ├── __init__.py
│   ├── gemini_analyzer.py   # Gemini API 통합
│   └── screen_capture.py    # 화면 캡처
│
├── physics/             # 물리 시뮬레이션
│   ├── __init__.py
│   ├── engine.py           # Pymunk 엔진 래퍼
│   └── simulator.py        # 시뮬레이션 실행
│
├── solver/              # 최적화 알고리즘
│   ├── __init__.py
│   ├── strategy.py         # 전략 평가
│   └── optimizer.py        # 위치 최적화
│
├── ui/                  # 사용자 인터페이스
│   ├── __init__.py
│   └── overlay.py          # 오버레이 윈도우
│
└── utils/               # 유틸리티
    ├── __init__.py
    ├── coordinate_mapper.py  # 좌표 변환
    └── state_detector.py     # 상태 감지
```

## 설정 커스터마이징

`config.py` 파일에서 다양한 파라미터를 조정할 수 있습니다:

### 물리 엔진 파라미터
```python
PHYSICS_GRAVITY = (0, 900)  # 중력
COIN_FRICTION = 0.5         # 마찰 계수
COIN_ELASTICITY = 0.3       # 탄성 계수
```

### Solver 설정
```python
SOLVER_ALGORITHM = "greedy"  # 알고리즘 선택
SOLVER_SAMPLE_STEP = 10      # 샘플링 간격 (픽셀)
```

### UI 설정
```python
OVERLAY_OPACITY = 0.7        # 투명도
GUIDE_LINE_COLOR = "#00FF00" # 가이드 라인 색상
```

## 동전 계층 구조

게임에서 사용되는 동전/지폐 계층:

1. **10원** → 2. **50원** → 3. **100원** → 4. **500원**
5. **1000원** → 6. **5000원** → 7. **10000원** → 8. **50000원**
9. **금괴(소)** → 10. **금괴(중)** → 11. **금괴(대)**

같은 종류의 동전 두 개가 합쳐지면 다음 레벨의 동전으로 진화합니다.

## 전략 알고리즘

프로그램은 다음과 같은 수박게임 전문 공략 전략을 사용합니다:

### 1. **좌우 정렬 전략**
- 작은 동전은 왼쪽, 큰 동전은 오른쪽에 배치
- 같은 크기의 동전을 옆에 나란히 배치하여 자연스럽게 합체 유도

### 2. **큰 동전 구석 배치**
- 큰 동전(지폐, 금괴)은 구석에 모아서 공간 효율성 극대화

### 3. **연쇄 합체 유도**
- 한 번의 합체가 다른 합체를 유발하도록 배치
- 높은 점수를 위한 콤보 생성

### 4. **블로킹 방지**
- 작은 동전이 큰 동전 사이에 끼지 않도록 주의
- 게임 오버의 주요 원인 방지

### 5. **높이 관리**
- 동전이 높이 쌓이지 않도록 관리
- 게임 오버 라인 초과 방지

## 평가 함수

게임 상태는 다음 기준으로 평가됩니다:

```python
Score = 
  + (큰 동전 개수 × 가중치)
  + (같은 동전 인접도)
  + (구석 배치 보너스)
  - (높이 페널티)
  - (블로킹 페널티)
```

## 문제 해결

### MuMu Player 창을 찾을 수 없음
- MuMu Player가 실행 중인지 확인
- 창 제목에 "MuMu"가 포함되어 있는지 확인
- `config.py`에서 `WINDOW_TITLE_PATTERN` 수정

### Gemini API 오류
- API 키가 올바르게 설정되었는지 확인
- 인터넷 연결 확인
- API 할당량 확인

### 오버레이가 표시되지 않음
- 일부 플랫폼에서는 투명 윈도우가 지원되지 않을 수 있음
- 관리자 권한으로 실행 시도

### 동전 인식 정확도가 낮음
- 게임 화면이 선명한지 확인
- 조명이 충분한지 확인
- Gemini 프롬프트 튜닝 (`vision/gemini_analyzer.py`)

## 성능 최적화

- **API 호출 최소화**: 정지 상태일 때만 분석
- **샘플링 간격 조정**: `SOLVER_SAMPLE_STEP` 증가 시 속도 향상 (정확도 감소)
- **시뮬레이션 시간 단축**: `SIMULATION_DURATION` 감소

## 라이선스

이 프로젝트는 교육 및 연구 목적으로 제작되었습니다.

## 기여

버그 리포트, 기능 제안, Pull Request를 환영합니다!

## 면책 조항

이 프로그램은 게임 공략을 위한 보조 도구이며, 게임의 자동 플레이를 수행하지 않습니다. 사용자는 여전히 수동으로 동전을 떨어뜨려야 합니다. 게임 서비스 약관을 준수하여 사용하시기 바랍니다.

---

**제작**: AI 기반 게임 공략 프로젝트  
**버전**: 1.0.0  
**최종 업데이트**: 2026-02-11
