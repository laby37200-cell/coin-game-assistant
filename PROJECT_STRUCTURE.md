# 동전게임 실시간 공략 가이드 - 프로젝트 구조

## 디렉토리 구조
```
coin_game_assistant/
├── main.py                 # 메인 실행 파일
├── config.py              # 설정 파일 (물리 파라미터, API 키 등)
├── requirements.txt       # 의존성 패키지 목록
├── README.md             # 사용 설명서
│
├── models/
│   ├── __init__.py
│   └── coin.py           # 동전 계층 구조 정의
│
├── vision/
│   ├── __init__.py
│   ├── gemini_analyzer.py    # Gemini API 통합
│   └── screen_capture.py     # 화면 캡처 모듈
│
├── physics/
│   ├── __init__.py
│   ├── engine.py             # Pymunk 물리 엔진 래퍼
│   └── simulator.py          # 물리 시뮬레이션 실행
│
├── solver/
│   ├── __init__.py
│   ├── strategy.py           # 전략 알고리즘
│   └── optimizer.py          # 최적 위치 계산
│
├── ui/
│   ├── __init__.py
│   └── overlay.py            # 오버레이 UI (tkinter)
│
└── utils/
    ├── __init__.py
    ├── coordinate_mapper.py  # 좌표계 변환
    └── state_detector.py     # 정지 상태 감지
```

## 모듈별 역할

### 1. models/coin.py
- 동전 종류 정의 (Enum)
- 각 동전의 물리적 속성 (반지름, 질량, 점수)
- 합체 규칙 정의

### 2. vision/gemini_analyzer.py
- Gemini 3.0 Flash Preview API 호출
- 이미지 분석 및 JSON 응답 파싱
- 동전 위치, 종류, 크기 추출

### 3. vision/screen_capture.py
- MuMu Player 창 찾기 (pygetwindow)
- 실시간 화면 캡처 (mss)
- 게임 영역 크롭

### 4. physics/engine.py
- Pymunk Space 초기화
- 물리 파라미터 설정 (중력, 마찰, 탄성)
- 바디 생성 및 관리

### 5. physics/simulator.py
- 현재 상태를 물리 공간에 복제
- 특정 위치에 동전 떨어뜨리기 시뮬레이션
- 결과 상태 평가

### 6. solver/strategy.py
- 수박게임 전략 구현
  - 좌우 정렬 전략
  - 큰 동전 구석 모으기
  - 연쇄 합체 유도
- 상태 평가 함수

### 7. solver/optimizer.py
- Monte Carlo / Greedy Search
- 가능한 모든 x좌표 테스트
- 최적 위치 반환

### 8. ui/overlay.py
- 투명 윈도우 생성 (tkinter)
- MuMu Player 위에 오버레이
- 가이드 라인 및 텍스트 표시

### 9. utils/coordinate_mapper.py
- 화면 좌표 ↔ 물리 엔진 좌표 변환
- 스케일링 및 오프셋 처리

### 10. utils/state_detector.py
- 프레임 비교로 정지 상태 감지
- 움직임 감지 알고리즘
- API 호출 타이밍 제어

## 데이터 흐름

```
[MuMu Player 화면]
    ↓ (screen_capture)
[스크린샷 이미지]
    ↓ (state_detector: 정지 상태 확인)
[Gemini API 호출]
    ↓ (gemini_analyzer)
[동전 상태 JSON]
    ↓ (coordinate_mapper)
[물리 엔진 좌표로 변환]
    ↓ (physics/engine: Digital Twin 생성)
[현재 상태 복제]
    ↓ (solver/optimizer: 모든 x좌표 시뮬레이션)
[최적 낙하 위치 계산]
    ↓ (coordinate_mapper: 화면 좌표로 역변환)
[오버레이 UI 업데이트]
    ↓
[사용자에게 가이드 표시]
```

## 실행 흐름

1. **초기화**
   - MuMu Player 창 찾기
   - Gemini API 클라이언트 초기화
   - 물리 엔진 설정
   - 오버레이 윈도우 생성

2. **메인 루프**
   ```python
   while True:
       # 1. 화면 캡처
       screenshot = capture_game_screen()
       
       # 2. 정지 상태 확인
       if not is_stable(screenshot):
           continue
       
       # 3. Gemini API로 분석
       coin_state = analyze_with_gemini(screenshot)
       
       # 4. 물리 엔진에 복제
       physics_world = create_digital_twin(coin_state)
       
       # 5. 최적 위치 계산
       best_x = find_optimal_drop_position(physics_world, next_coin)
       
       # 6. 오버레이 업데이트
       update_overlay(best_x)
       
       # 7. 대기 (동전이 떨어질 때까지)
       wait_for_next_turn()
   ```

3. **종료**
   - 사용자 입력 (ESC 키 등)
   - 리소스 정리
