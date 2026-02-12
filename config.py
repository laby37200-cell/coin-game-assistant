"""
동전게임 공략 가이드 설정 파일

물리 엔진 파라미터, API 설정, UI 설정 등을 관리합니다.
"""

import os
from typing import Tuple


# ==================== API 설정 ====================

# Gemini API 설정
GEMINI_API_KEY = "AIzaSyDxPwUrvGHzmghDomx9q-NX9gDtZDbxNWQ"
GEMINI_MODEL = "gemini-2.5-flash"  # 사용할 모델명 (최신 안정 Flash)

# API 호출 제한
MAX_API_CALLS_PER_MINUTE = 10  # 분당 최대 호출 횟수
API_TIMEOUT = 10  # API 타임아웃 (초)


# ==================== 화면 캡처 설정 ====================

# MuMu Player 창 설정
WINDOW_TITLE_PATTERN = "MuMuPlayer"  # MuMu Player 창 제목 패턴
CAPTURE_FPS = 2  # 초당 캡처 프레임 수 (너무 높으면 CPU 부하)

# 게임 영역 설정 (자동 감지 실패 시 사용)
GAME_AREA_OFFSET_X = 0  # 게임 영역 x 오프셋
GAME_AREA_OFFSET_Y = 0  # 게임 영역 y 오프셋
GAME_AREA_WIDTH = 600   # 게임 영역 너비
GAME_AREA_HEIGHT = 800  # 게임 영역 높이


# ==================== 물리 엔진 설정 ====================

# Pymunk 물리 파라미터 (수박게임과 유사하게 튜닝)
PHYSICS_GRAVITY = (0, -900)  # 중력 (x, y) - y축 위 방향이 양수, 아래 방향이 음수
PHYSICS_DAMPING = 0.95      # 감쇠 계수 (0~1, 1에 가까울수록 에너지 보존)
PHYSICS_ITERATIONS = 10     # 물리 시뮬레이션 반복 횟수 (높을수록 정확하지만 느림)

# 동전 물리 속성
COIN_FRICTION = 0.5         # 마찰 계수
COIN_ELASTICITY = 0.3       # 탄성 계수 (0=완전 비탄성, 1=완전 탄성)
COIN_DENSITY = 1.0          # 밀도

# 벽 및 바닥 설정
WALL_FRICTION = 0.6
WALL_ELASTICITY = 0.2

# 시뮬레이션 설정
SIMULATION_TIME_STEP = 1/60  # 시뮬레이션 타임스텝 (초)
SIMULATION_DURATION = 3.0    # 각 시뮬레이션 지속 시간 (초)
STABILITY_THRESHOLD = 0.1    # 안정 상태 판단 임계값 (속도)


# ==================== 좌표계 변환 설정 ====================

# 화면 좌표계 → 물리 엔진 좌표계 변환
# 화면: 왼쪽 상단이 (0, 0), 오른쪽 하단이 (width, height)
# 물리: 왼쪽 하단이 (0, 0), 오른쪽 상단이 (width, height)
PHYSICS_SCALE = 1.0  # 스케일 팩터 (필요시 조정)


# ==================== 상태 감지 설정 ====================

# 정지 상태 감지 파라미터
STABILITY_CHECK_FRAMES = 5      # 연속으로 확인할 프레임 수
STABILITY_PIXEL_THRESHOLD = 100  # 프레임 간 차이 픽셀 임계값
STABILITY_WAIT_TIME = 0.5       # 안정 상태 확인 대기 시간 (초)


# ==================== Solver 설정 ====================

# 최적화 알고리즘 설정
SOLVER_ALGORITHM = "greedy"  # "greedy" 또는 "monte_carlo"
SOLVER_SAMPLE_STEP = 10      # x좌표 샘플링 간격 (픽셀)
SOLVER_LOOKAHEAD_DEPTH = 2   # Look-ahead 깊이 (1=현재만, 2=다음 턴까지)

# 평가 함수 가중치
WEIGHT_LARGE_COIN = 10.0      # 큰 동전 보너스
WEIGHT_ADJACENCY = 5.0        # 같은 동전 인접도 보너스
WEIGHT_HEIGHT_PENALTY = -2.0  # 높이 페널티
WEIGHT_CORNER_BONUS = 3.0     # 구석 배치 보너스
WEIGHT_BLOCKING_PENALTY = -20.0  # 블로킹 페널티 (작은 동전이 큰 동전 사이에 낌)

# 전략 설정
STRATEGY_PREFER_CORNER = True   # 큰 동전을 구석에 배치
STRATEGY_LEFT_TO_RIGHT = True   # 좌→우 정렬 전략 사용


# ==================== UI 오버레이 설정 ====================

# 오버레이 윈도우 설정
OVERLAY_OPACITY = 0.7        # 투명도 (0~1)
OVERLAY_UPDATE_INTERVAL = 500  # 업데이트 간격 (밀리초)

# 가이드 라인 스타일
GUIDE_LINE_COLOR = "#00FF00"  # 녹색
GUIDE_LINE_WIDTH = 3
GUIDE_LINE_STYLE = "solid"    # "solid" 또는 "dashed"

# 텍스트 스타일
GUIDE_TEXT_COLOR = "#FFFFFF"  # 흰색
GUIDE_TEXT_SIZE = 16
GUIDE_TEXT_FONT = "Arial"
GUIDE_TEXT_SHADOW = True

# 가이드 메시지
GUIDE_MESSAGE_TEMPLATE = "여기에 떨어트리세요!\n예상 점수: {score}"


# ==================== 디버그 설정 ====================

DEBUG_MODE = True             # 디버그 모드 활성화
DEBUG_SHOW_PHYSICS = False    # 물리 시뮬레이션 시각화
DEBUG_SAVE_SCREENSHOTS = False  # 스크린샷 저장
DEBUG_LOG_LEVEL = "DEBUG"     # 로그 레벨: DEBUG, INFO, WARNING, ERROR


# ==================== 게임 규칙 설정 ====================

# 게임 영역 크기 (픽셀, 자동 감지 후 업데이트됨)
GAME_WIDTH = 600
GAME_HEIGHT = 800

# 게임 오버 라인 (상단으로부터의 거리)
GAME_OVER_LINE_Y = 100

# 드롭 가능 영역
DROP_MIN_X = 50   # 최소 x 좌표
DROP_MAX_X = 550  # 최대 x 좌표
DROP_Y = 50       # 드롭 y 좌표 (상단)


# ==================== 유틸리티 함수 ====================

def validate_config():
    """설정 유효성 검사"""
    errors = []
    
    if not GEMINI_API_KEY:
        errors.append("GEMINI_API_KEY가 설정되지 않았습니다. 환경 변수를 확인하세요.")
    
    if SIMULATION_TIME_STEP <= 0:
        errors.append("SIMULATION_TIME_STEP은 양수여야 합니다.")
    
    if not (0 <= OVERLAY_OPACITY <= 1):
        errors.append("OVERLAY_OPACITY는 0과 1 사이여야 합니다.")
    
    if SOLVER_SAMPLE_STEP <= 0:
        errors.append("SOLVER_SAMPLE_STEP은 양수여야 합니다.")
    
    return errors


def print_config():
    """현재 설정 출력 (디버깅용)"""
    print("=== 동전게임 공략 가이드 설정 ===\n")
    print(f"[API 설정]")
    print(f"  모델: {GEMINI_MODEL}")
    print(f"  API 키: {'설정됨' if GEMINI_API_KEY else '미설정'}")
    print()
    print(f"[물리 엔진]")
    print(f"  중력: {PHYSICS_GRAVITY}")
    print(f"  마찰: {COIN_FRICTION}")
    print(f"  탄성: {COIN_ELASTICITY}")
    print()
    print(f"[Solver]")
    print(f"  알고리즘: {SOLVER_ALGORITHM}")
    print(f"  샘플링 간격: {SOLVER_SAMPLE_STEP}px")
    print()
    print(f"[디버그]")
    print(f"  디버그 모드: {DEBUG_MODE}")
    print(f"  로그 레벨: {DEBUG_LOG_LEVEL}")
    print()


if __name__ == "__main__":
    print_config()
    
    errors = validate_config()
    if errors:
        print("\n[설정 오류]")
        for error in errors:
            print(f"  ❌ {error}")
    else:
        print("✅ 모든 설정이 유효합니다.")
