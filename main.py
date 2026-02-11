"""
동전게임 실시간 공략 가이드 - 메인 프로그램

MuMu Player에서 실행되는 동전게임을 실시간으로 분석하여
최적의 낙하 위치를 오버레이로 표시합니다.
"""

import os
import sys
import time
import logging
from typing import Optional

# 프로젝트 루트를 PYTHONPATH에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from models.coin import Coin, CoinType
from vision.screen_capture import ScreenCapture
from vision.gemini_analyzer import GeminiAnalyzer
from physics.simulator import PhysicsSimulator
from solver.optimizer import PositionOptimizer
from ui.overlay import OverlayWindow
from utils.state_detector import StateDetector
from utils.coordinate_mapper import CoordinateMapper


# 로깅 설정
logging.basicConfig(
    level=getattr(logging, config.DEBUG_LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CoinGameAssistant:
    """동전게임 공략 가이드 메인 클래스"""
    
    def __init__(self):
        """초기화"""
        logger.info("=== 동전게임 공략 가이드 시작 ===")
        
        # 설정 검증
        errors = config.validate_config()
        if errors:
            for error in errors:
                logger.error(error)
            raise ValueError("설정 오류가 있습니다. config.py를 확인하세요.")
        
        # 컴포넌트 초기화
        self.screen_capture: Optional[ScreenCapture] = None
        self.gemini_analyzer: Optional[GeminiAnalyzer] = None
        self.physics_simulator: Optional[PhysicsSimulator] = None
        self.position_optimizer: Optional[PositionOptimizer] = None
        self.overlay_window: Optional[OverlayWindow] = None
        self.state_detector: Optional[StateDetector] = None
        self.coordinate_mapper: Optional[CoordinateMapper] = None
        
        # 게임 상태
        self.game_width = config.GAME_WIDTH
        self.game_height = config.GAME_HEIGHT
        self.running = False
    
    def initialize(self) -> bool:
        """모든 컴포넌트 초기화"""
        try:
            # 1. 화면 캡처 초기화
            logger.info("화면 캡처 초기화 중...")
            self.screen_capture = ScreenCapture(config.WINDOW_TITLE_PATTERN)
            
            if not self.screen_capture.find_window():
                logger.error("MuMu Player 창을 찾을 수 없습니다.")
                logger.error("MuMu Player를 실행하고 동전게임을 시작한 후 다시 시도하세요.")
                return False
            
            # 게임 영역 크기 업데이트
            dimensions = self.screen_capture.get_game_dimensions()
            if dimensions:
                self.game_width, self.game_height = dimensions
                logger.info(f"게임 영역: {self.game_width}x{self.game_height}")
            
            # 2. Gemini 분석기 초기화
            logger.info("Gemini 분석기 초기화 중...")
            self.gemini_analyzer = GeminiAnalyzer(
                api_key=config.GEMINI_API_KEY,
                model_name=config.GEMINI_MODEL
            )
            
            # 3. 물리 시뮬레이터 초기화
            logger.info("물리 시뮬레이터 초기화 중...")
            self.physics_simulator = PhysicsSimulator(
                game_width=self.game_width,
                game_height=self.game_height,
                time_step=config.SIMULATION_TIME_STEP,
                simulation_duration=config.SIMULATION_DURATION
            )
            
            # 4. 위치 최적화기 초기화
            logger.info("위치 최적화기 초기화 중...")
            self.position_optimizer = PositionOptimizer(
                game_width=self.game_width,
                game_height=self.game_height,
                algorithm=config.SOLVER_ALGORITHM,
                sample_step=config.SOLVER_SAMPLE_STEP,
                lookahead_depth=config.SOLVER_LOOKAHEAD_DEPTH
            )
            
            # 5. 오버레이 윈도우 초기화
            logger.info("오버레이 윈도우 초기화 중...")
            window_area = self.screen_capture.game_area
            self.overlay_window = OverlayWindow(
                window_x=window_area['left'],
                window_y=window_area['top'],
                window_width=window_area['width'],
                window_height=window_area['height'],
                opacity=config.OVERLAY_OPACITY,
                line_color=config.GUIDE_LINE_COLOR,
                line_width=config.GUIDE_LINE_WIDTH,
                text_color=config.GUIDE_TEXT_COLOR,
                text_size=config.GUIDE_TEXT_SIZE
            )
            self.overlay_window.create_window()
            
            # 6. 상태 감지기 초기화
            logger.info("상태 감지기 초기화 중...")
            self.state_detector = StateDetector(
                check_frames=config.STABILITY_CHECK_FRAMES,
                pixel_threshold=config.STABILITY_PIXEL_THRESHOLD,
                wait_time=config.STABILITY_WAIT_TIME
            )
            
            # 7. 좌표 변환기 초기화
            self.coordinate_mapper = CoordinateMapper(
                screen_width=self.game_width,
                screen_height=self.game_height,
                physics_width=self.game_width,
                physics_height=self.game_height,
                scale=config.PHYSICS_SCALE
            )
            
            logger.info("✅ 모든 컴포넌트 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"초기화 실패: {e}", exc_info=True)
            return False
    
    def run(self):
        """메인 루프 실행"""
        self.running = True
        logger.info("메인 루프 시작")
        
        # 시작 메시지
        self.overlay_window.show_message("동전게임 공략 가이드 시작!\n게임이 정지 상태가 되면 분석합니다.", duration=3000)
        
        try:
            while self.running:
                # 1. 화면 캡처
                screenshot = self.screen_capture.capture()
                
                if screenshot is None:
                    logger.warning("화면 캡처 실패")
                    time.sleep(1)
                    continue
                
                # 2. 상태 감지 (정지 상태 확인)
                self.state_detector.add_frame(screenshot)
                
                if not self.state_detector.is_stable():
                    # 움직이는 중이면 대기
                    self.overlay_window.update()
                    time.sleep(0.2)
                    continue
                
                logger.info("정지 상태 감지 - 분석 시작")
                
                # 3. Gemini API로 화면 분석
                pil_image = self.screen_capture.capture_pil()
                
                if pil_image is None:
                    logger.warning("PIL 이미지 변환 실패")
                    time.sleep(1)
                    continue
                
                coins, next_coin_type, game_area = self.gemini_analyzer.analyze_and_extract(pil_image)
                
                if not coins:
                    logger.warning("동전을 감지하지 못했습니다.")
                    self.overlay_window.show_message("동전을 감지하지 못했습니다.", duration=2000)
                    time.sleep(2)
                    continue
                
                logger.info(f"동전 {len(coins)}개 감지, 다음 동전: {next_coin_type.display_name if next_coin_type else '알 수 없음'}")
                
                # 4. 다음 동전이 없으면 기본값 사용
                if not next_coin_type:
                    next_coin_type = CoinType.COIN_10  # 기본값
                    logger.warning("다음 동전 정보 없음, 기본값 사용")
                
                # 5. 최적 위치 계산
                logger.info("최적 위치 계산 중...")
                best_x, best_score, details = self.position_optimizer.find_optimal_position(
                    coins,
                    next_coin_type
                )
                
                logger.info(f"최적 위치: x={best_x:.1f}, 예상 점수={best_score:.1f}")
                
                # 6. 오버레이 업데이트
                self.overlay_window.update_guide(best_x, best_score)
                
                # 7. 다음 턴까지 대기
                logger.info("다음 턴 대기 중...")
                time.sleep(2)
                
                # 상태 감지기 리셋
                self.state_detector.reset()
                
                # UI 업데이트
                self.overlay_window.update()
                
        except KeyboardInterrupt:
            logger.info("사용자에 의해 중단됨")
        except Exception as e:
            logger.error(f"메인 루프 오류: {e}", exc_info=True)
        finally:
            self.cleanup()
    
    def cleanup(self):
        """리소스 정리"""
        logger.info("리소스 정리 중...")
        
        if self.overlay_window:
            self.overlay_window.close()
        
        self.running = False
        logger.info("프로그램 종료")


def main():
    """메인 함수"""
    print("""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║         동전게임 실시간 공략 가이드                          ║
║                                                           ║
║   MuMu Player에서 동전게임을 실행한 후 이 프로그램을         ║
║   시작하세요. 게임 화면 위에 최적의 낙하 위치가 표시됩니다.   ║
║                                                           ║
║   종료: Ctrl+C 또는 오버레이 창 닫기                        ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    # CoinGameAssistant 인스턴스 생성
    assistant = CoinGameAssistant()
    
    # 초기화
    if not assistant.initialize():
        print("\n❌ 초기화 실패. 로그를 확인하세요.")
        return 1
    
    print("\n✅ 초기화 완료! 게임 분석을 시작합니다...\n")
    
    # 메인 루프 실행
    assistant.run()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
