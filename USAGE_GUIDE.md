# 동전게임 공략 가이드 - 사용 가이드

## 빠른 시작 (Quick Start)

### 1단계: 환경 설정

```bash
# 1. Python 3.10+ 설치 확인
python --version

# 2. 프로젝트 디렉토리로 이동
cd coin_game_assistant

# 3. 의존성 설치
pip install -r requirements.txt

# 4. Gemini API 키 설정 (필수!)
export GEMINI_API_KEY="your-api-key-here"
```

### 2단계: 게임 준비

1. **MuMu Player** 실행
2. **동전게임** 앱 실행
3. 게임을 플레이 가능한 상태로 만들기

### 3단계: 프로그램 실행

```bash
python main.py
```

### 4단계: 사용

- 게임이 정지 상태가 되면 자동으로 분석
- 녹색 수직선이 최적 위치 표시
- 해당 위치에 동전 떨어뜨리기
- 반복!

---

## 상세 사용법

### API 키 발급 방법

1. [Google AI Studio](https://makersuite.google.com/app/apikey) 접속
2. Google 계정으로 로그인
3. "Create API Key" 클릭
4. 생성된 API 키 복사
5. 환경 변수에 설정

**Windows PowerShell:**
```powershell
$env:GEMINI_API_KEY="AIza..."
```

**Linux/Mac:**
```bash
export GEMINI_API_KEY="AIza..."
```

**영구 설정 (Linux/Mac):**
```bash
echo 'export GEMINI_API_KEY="AIza..."' >> ~/.bashrc
source ~/.bashrc
```

### 프로그램 실행 옵션

#### 기본 실행
```bash
python main.py
```

#### 디버그 모드
`config.py`에서 다음 설정 변경:
```python
DEBUG_MODE = True
DEBUG_LOG_LEVEL = "DEBUG"
```

#### 스크린샷 저장 모드
```python
DEBUG_SAVE_SCREENSHOTS = True
```

### 게임 플레이 팁

#### 1. 프로그램 시작 타이밍
- 게임이 **완전히 로드**된 후 프로그램 시작
- 첫 동전이 떨어지기 **전**에 시작하는 것이 좋음

#### 2. 정지 상태 확인
- 프로그램은 **정지 상태**일 때만 분석
- 동전이 움직이는 중에는 대기
- 약 0.5초 정지 후 분석 시작

#### 3. 가이드 라인 해석
- **녹색 수직선**: 최적의 x 좌표
- **텍스트**: 예상 점수
- 점수가 높을수록 좋은 위치

#### 4. 수동 조정
- 가이드는 **참고용**
- 상황에 따라 약간 조정 가능
- 게임 감각과 병행 사용 권장

---

## 설정 커스터마이징

### 물리 엔진 튜닝

게임의 물리 특성과 맞지 않으면 `config.py`에서 조정:

```python
# 중력 (y 값이 클수록 빠르게 떨어짐)
PHYSICS_GRAVITY = (0, 900)

# 마찰력 (0~1, 높을수록 잘 안 굴러감)
COIN_FRICTION = 0.5

# 탄성 (0~1, 높을수록 잘 튕김)
COIN_ELASTICITY = 0.3
```

### Solver 알고리즘 조정

```python
# 알고리즘 선택
SOLVER_ALGORITHM = "greedy"  # 또는 "monte_carlo"

# 샘플링 간격 (작을수록 정확, 느림)
SOLVER_SAMPLE_STEP = 10  # 픽셀

# Look-ahead 깊이 (현재는 1만 지원)
SOLVER_LOOKAHEAD_DEPTH = 1
```

### 평가 함수 가중치 조정

`config.py`에서 전략 가중치 변경:

```python
# 큰 동전 보너스 (높을수록 큰 동전 선호)
WEIGHT_LARGE_COIN = 10.0

# 인접도 보너스 (같은 동전끼리 모으기)
WEIGHT_ADJACENCY = 5.0

# 높이 페널티 (높이 쌓이는 것 방지)
WEIGHT_HEIGHT_PENALTY = -2.0

# 구석 배치 보너스
WEIGHT_CORNER_BONUS = 3.0

# 블로킹 페널티 (작은 동전이 큰 동전 사이에 끼임)
WEIGHT_BLOCKING_PENALTY = -20.0
```

### UI 커스터마이징

```python
# 오버레이 투명도 (0~1)
OVERLAY_OPACITY = 0.7

# 가이드 라인 색상
GUIDE_LINE_COLOR = "#00FF00"  # 녹색

# 가이드 라인 두께
GUIDE_LINE_WIDTH = 3

# 텍스트 색상
GUIDE_TEXT_COLOR = "#FFFFFF"  # 흰색

# 텍스트 크기
GUIDE_TEXT_SIZE = 16
```

---

## 문제 해결 (Troubleshooting)

### 문제 1: "MuMu Player 창을 찾을 수 없습니다"

**원인:**
- MuMu Player가 실행되지 않음
- 창 제목이 "MuMu"를 포함하지 않음

**해결:**
1. MuMu Player가 실행 중인지 확인
2. `config.py`에서 `WINDOW_TITLE_PATTERN` 수정:
   ```python
   WINDOW_TITLE_PATTERN = "MuMu"  # 창 제목의 일부
   ```
3. 다른 에뮬레이터 사용 시 해당 창 제목으로 변경

### 문제 2: "GEMINI_API_KEY가 설정되지 않았습니다"

**원인:**
- 환경 변수가 설정되지 않음

**해결:**
```bash
# 환경 변수 설정
export GEMINI_API_KEY="your-api-key-here"

# 또는 config.py에서 직접 설정
GEMINI_API_KEY = "your-api-key-here"
```

### 문제 3: "동전을 감지하지 못했습니다"

**원인:**
- 게임 화면이 흐림
- Gemini API가 동전을 인식하지 못함

**해결:**
1. 게임 화면 해상도 확인
2. 조명 밝기 조정
3. 게임이 완전히 로드되었는지 확인
4. `vision/gemini_analyzer.py`의 프롬프트 튜닝

### 문제 4: 오버레이가 표시되지 않음

**원인:**
- 플랫폼이 투명 윈도우를 지원하지 않음
- 권한 문제

**해결:**
1. 관리자 권한으로 실행
2. `config.py`에서 투명도 조정:
   ```python
   OVERLAY_OPACITY = 1.0  # 완전 불투명
   ```

### 문제 5: 프로그램이 느림

**원인:**
- 샘플링 간격이 너무 작음
- 시뮬레이션 시간이 너무 김

**해결:**
```python
# config.py에서 조정
SOLVER_SAMPLE_STEP = 20  # 기본값 10에서 증가
SIMULATION_DURATION = 2.0  # 기본값 3.0에서 감소
```

### 문제 6: API 할당량 초과

**원인:**
- Gemini API 무료 할당량 초과

**해결:**
1. API 호출 빈도 줄이기:
   ```python
   STABILITY_WAIT_TIME = 1.0  # 기본값 0.5에서 증가
   ```
2. 유료 플랜 고려

---

## 고급 사용법

### 1. 커스텀 동전 계층 구조

게임이 다른 동전 구조를 사용하면 `models/coin.py` 수정:

```python
class CoinType(Enum):
    # 새로운 동전 추가
    COIN_10 = (1, "10원", 15, 10, (255, 200, 100))
    # (레벨, 이름, 반지름, 점수, RGB색상)
```

### 2. 전략 알고리즘 개발

`solver/strategy.py`에서 새로운 평가 함수 추가:

```python
def _evaluate_custom_strategy(self, coins: List[Coin]) -> float:
    """커스텀 전략 평가"""
    score = 0.0
    # 여기에 로직 구현
    return score
```

### 3. 물리 시뮬레이션 시각화

디버그 모드에서 물리 시뮬레이션 결과 확인:

```python
DEBUG_SHOW_PHYSICS = True  # config.py
```

### 4. 배치 테스트

여러 위치를 한 번에 비교:

```python
from solver.optimizer import PositionOptimizer

optimizer = PositionOptimizer(600, 800)
results = optimizer.compare_positions(
    current_coins,
    drop_coin_type,
    [100, 200, 300, 400, 500]
)

for x, score, details in results:
    print(f"x={x}: score={score}")
```

---

## 성능 벤치마크

### 일반적인 성능

- **화면 캡처**: ~10ms
- **Gemini API 호출**: ~1-3초
- **물리 시뮬레이션**: ~0.5초 (샘플링 30개)
- **전체 분석 주기**: ~3-5초

### 최적화 팁

1. **샘플링 간격 증가**: 속도 ↑, 정확도 ↓
2. **시뮬레이션 시간 단축**: 속도 ↑, 안정성 ↓
3. **정지 상태 대기 시간 증가**: API 호출 ↓, 반응 속도 ↓

---

## FAQ

### Q1: 게임을 자동으로 플레이하나요?
**A:** 아니요. 이 프로그램은 **가이드만 제공**하며, 사용자가 직접 동전을 떨어뜨려야 합니다.

### Q2: 어떤 게임에서 사용할 수 있나요?
**A:** '콘텐츠페이 동전게임' 및 유사한 수박게임 변형판에서 사용 가능합니다.

### Q3: API 비용이 얼마나 드나요?
**A:** Gemini API는 무료 할당량이 있으며, 초과 시 유료입니다. 정지 상태에서만 호출하므로 비용이 크지 않습니다.

### Q4: 정확도는 어느 정도인가요?
**A:** 물리 엔진 파라미터 튜닝에 따라 다르지만, 일반적으로 **70-90%** 정확도를 보입니다.

### Q5: 다른 에뮬레이터에서도 작동하나요?
**A:** 네, BlueStacks, NoxPlayer 등 다른 에뮬레이터에서도 작동합니다. `WINDOW_TITLE_PATTERN`만 수정하면 됩니다.

### Q6: 모바일에서 사용할 수 있나요?
**A:** 현재 버전은 PC에서 에뮬레이터를 사용하는 경우만 지원합니다.

---

## 지원 및 문의

- **버그 리포트**: GitHub Issues
- **기능 제안**: GitHub Discussions
- **문의**: 프로젝트 README 참조

---

**즐거운 게임 되세요! 🎮**
