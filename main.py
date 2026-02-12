"""
동전게임 실시간 공략 가이드 - 메인 프로그램

MuMu Player에서 실행되는 동전게임을 실시간으로 분석하여
최적의 낙하 위치를 오버레이로 표시합니다.
"""

import os
import sys
import re
import time
import logging
import threading
from typing import Optional

# Windows 콘솔 UTF-8 출력 보장 (exe 실행 시 인코딩 오류 방지)
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# 프로젝트 루트를 PYTHONPATH에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from models.coin import Coin, CoinType
from vision.screen_capture import ScreenCapture
from vision.gemini_analyzer import GeminiAnalyzer
from physics.simulator import PhysicsSimulator
from solver.optimizer import PositionOptimizer
from ui.overlay import OverlayWindow, GuideInfo
from utils.state_detector import StateDetector
from utils.coordinate_mapper import CoordinateMapper
from ai.auto_tuner import AutoTuner
from ai.feedback_loop import FeedbackLoop
try:
    from solver.gpu_physics import GPUPhysicsBatch, PhysicsParams, PhysicsCalibrator
    from solver.mcts_engine import MCTSEngine
    _HAS_MCTS = True
except ImportError:
    _HAS_MCTS = False
    GPUPhysicsBatch = None
    PhysicsCalibrator = None
    MCTSEngine = None


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
        self.auto_tuner: Optional[AutoTuner] = None
        self.feedback_loop: Optional[FeedbackLoop] = None
        self.gpu_physics: Optional[GPUPhysicsBatch] = None
        self.mcts_engine: Optional[MCTSEngine] = None
        self.calibrator: Optional[PhysicsCalibrator] = None
        
        # 게임 상태
        self.game_width = config.GAME_WIDTH
        self.game_height = config.GAME_HEIGHT
        self.running = False
        self._game_score = 0           # 실제 인게임 점수 추적
        self._wall_left = 105.0
        self._wall_right = 435.0
        self._ceiling_y = 200.0
        self._floor_y = 870.0
        self._load_bounds()  # 저장된 경계값 로드
    
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
                model_name=config.GEMINI_MODEL,
                max_calls_per_minute=config.MAX_API_CALLS_PER_MINUTE,
            )

            # 3. 좌표 변환기 초기화 (화면(y-down) -> 물리(y-up))
            self.coordinate_mapper = CoordinateMapper(
                screen_width=self.game_width,
                screen_height=self.game_height,
                physics_width=self.game_width,
                physics_height=self.game_height,
                scale=config.PHYSICS_SCALE,
            )

            physics_params = {
                'gravity': config.PHYSICS_GRAVITY,
                'damping': config.PHYSICS_DAMPING,
                'iterations': config.PHYSICS_ITERATIONS,
                'coin_friction': config.COIN_FRICTION,
                'coin_elasticity': config.COIN_ELASTICITY,
                'wall_friction': config.WALL_FRICTION,
                'wall_elasticity': config.WALL_ELASTICITY,
            }
            
            # 4. 물리 시뮬레이터 초기화
            logger.info("물리 시뮬레이터 초기화 중...")
            self.physics_simulator = PhysicsSimulator(
                game_width=self.game_width,
                game_height=self.game_height,
                coordinate_mapper=self.coordinate_mapper,
                time_step=config.SIMULATION_TIME_STEP,
                simulation_duration=config.SIMULATION_DURATION,
                **physics_params,
            )
            
            # 5. 위치 최적화기 초기화
            logger.info("위치 최적화기 초기화 중...")
            self.position_optimizer = PositionOptimizer(
                game_width=self.game_width,
                game_height=self.game_height,
                algorithm=config.SOLVER_ALGORITHM,
                sample_step=config.SOLVER_SAMPLE_STEP,
                lookahead_depth=config.SOLVER_LOOKAHEAD_DEPTH,
                coordinate_mapper=self.coordinate_mapper,
                physics_params=physics_params,
            )
            
            # 6. 오버레이 윈도우 초기화
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
            self.overlay_window.update()  # 창을 즉시 화면에 표시

            # 오버레이에 초기 벽/천장/바닥 경계 표시
            self.overlay_window.update_bounds(
                self._wall_left, self._wall_right,
                self._ceiling_y, self._floor_y)
            self.overlay_window.set_bounds_callback(self._on_bounds_adjusted)
            self.overlay_window.set_analyze_callback(self._on_manual_analyze)

            # 7. 상태 감지기 초기화
            logger.info("상태 감지기 초기화 중...")
            self.state_detector = StateDetector(
                check_frames=config.STABILITY_CHECK_FRAMES,
                pixel_threshold=config.STABILITY_PIXEL_THRESHOLD,
                wait_time=config.STABILITY_WAIT_TIME
            )
            
            # 8. 자가 피드백 루프 초기화
            logger.info("자가 피드백 루프 초기화 중...")
            try:
                self.auto_tuner = AutoTuner(api_key=config.GEMINI_API_KEY)
                self.feedback_loop = FeedbackLoop(
                    auto_tuner=self.auto_tuner,
                    simulator=self.physics_simulator,
                    min_accuracy=0.8,
                    max_iterations=5
                )
                logger.info("자가 피드백 루프 초기화 완료")
            except Exception as e:
                logger.warning(f"피드백 루프 초기화 실패 (계속 진행): {e}")

            # 9. GPU 물리 시뮬레이션 + MCTS 엔진 초기화
            if _HAS_MCTS:
                logger.info("GPU MCTS 엔진 초기화 중...")
                try:
                    self.gpu_physics = GPUPhysicsBatch()
                    self.calibrator = PhysicsCalibrator(self.gpu_physics)
                    calib_path = os.path.join(os.path.dirname(__file__), 'calibration.json')
                    self.calibrator.load(calib_path)
                    self.mcts_engine = MCTSEngine(
                        gpu_physics=self.gpu_physics,
                        wall_left=self._wall_left,
                        wall_right=self._wall_right,
                        floor_y=self._floor_y,
                        ceiling_y=self._ceiling_y,
                        num_positions=14,
                        max_depth=15,
                        time_budget_s=6.0,
                        max_iterations=500,
                    )
                    logger.info(f"GPU MCTS 초기화 완료 (device={self.gpu_physics.device})")
                    # Pre-warm the simulation worker pool
                    try:
                        from solver.fast_sim import warmup_pool
                        warmup_pool()
                    except Exception:
                        pass
                except Exception as e:
                    logger.warning(f"GPU MCTS 초기화 실패 — 기본 optimizer 사용: {e}")
                    self.mcts_engine = None
            else:
                logger.info("torch 미설치 — MCTS 비활성, 기본 optimizer 사용")

            logger.info("All components initialized")
            return True
            
        except Exception as e:
            logger.error(f"초기화 실패: {e}", exc_info=True)
            return False
    
    def run(self):
        """메인 루프 실행 (Gemini/시뮬레이션은 백그라운드 스레드에서 실행)"""
        self.running = True
        logger.info("메인 루프 시작")

        # 백그라운드 분석 결과를 공유하기 위한 변수
        self._analysis_lock = threading.Lock()
        self._analysis_busy = False
        self._manual_trigger = False      # 수동 분석 요청 플래그
        self._last_result = None          # (best_x, best_score, info_str)
        self._last_before_state = None    # 피드백용: 드롭 전 상태
        self._last_drop_coin = None       # 피드백용: 드롭한 동전
        self._last_drop_x = None          # 피드백용: 드롭 위치
        self._last_predicted = None       # 피드백용: 예측 결과

        # 시작 메시지
        self.overlay_window.show_message("Coin Game Assistant\n[Tab] 분석  [Ctrl+Tab] 자동모드", duration=5000)

        try:
            while self.running:
                # tkinter 이벤트 처리 (항상 호출 — 검은화면 방지)
                self.overlay_window.update()

                if not self.overlay_window.is_open():
                    logger.info("오버레이 창이 닫혔습니다.")
                    break

                # 1. 화면 캡처
                screenshot = self.screen_capture.capture()
                if screenshot is None:
                    time.sleep(0.5)
                    continue

                # 2. 상태 감지
                self.state_detector.add_frame(screenshot)

                if not self.state_detector.is_stable():
                    time.sleep(max(0.01, 1.0 / max(1, config.CAPTURE_FPS)))
                    continue

                # 3. 백그라운드 분석 결과가 있으면 오버레이 업데이트
                with self._analysis_lock:
                    if self._last_result is not None:
                        guides, best_score, info_str = self._last_result
                        self._last_result = None
                        if isinstance(guides, list) and guides:
                            self.overlay_window.update_guides(guides)
                            self.overlay_window.update_status(info_str)
                            best_x = guides[0].x
                            logger.info(f"Guides: {len(guides)} lines, best x={best_x:.1f} | {info_str}")
                        else:
                            self.overlay_window.update_guide(best_score, 0)

                # 4. 분석 시작 (수동 모드: Tab 누를 때만 / 자동 모드: 항상)
                with self._analysis_lock:
                    busy = self._analysis_busy
                    trigger = self._manual_trigger
                    self._manual_trigger = False

                should_analyze = (not busy) and (
                    self.overlay_window.auto_mode or trigger
                )

                if should_analyze:
                    pil_image = self.screen_capture.capture_pil()
                    if pil_image is not None:
                        with self._analysis_lock:
                            self._analysis_busy = True
                        mode = "자동" if self.overlay_window.auto_mode else "수동"
                        self.overlay_window.update_status(f"[{mode}] 분석 시작...")
                        self.overlay_window.update_progress(0.0)
                        t = threading.Thread(target=self._background_analyze, args=(pil_image,), daemon=True)
                        t.start()

                time.sleep(max(0.01, 1.0 / max(1, config.CAPTURE_FPS)))

        except KeyboardInterrupt:
            logger.info("사용자에 의해 중단됨")
        except Exception as e:
            logger.error(f"메인 루프 오류: {e}", exc_info=True)
        finally:
            self.cleanup()

    def _progress_cb(self, progress: float, status: str):
        """MCTS 진행도 콜백 — 오버레이에 실시간 가이드 + 진행도 표시"""
        if not self.overlay_window:
            return
        self.overlay_window.update_progress(progress)
        self.overlay_window.update_status(status)

        # Parse intermediate candidates from status: "MCTS: x=200(85%) | x=300(60%) ..."
        if 'MCTS:' in status and 'x=' in status:
            pattern = re.compile(r'x=(\d+)\((\d+)%\)')
            matches = pattern.findall(status)
            if matches:
                radius = getattr(self, '_current_radius', 20.0)
                guides = []
                for rank, (xs, cs) in enumerate(matches):
                    gx = self._clamp_x(float(xs), radius)
                    conf = float(cs) / 100.0
                    guides.append(GuideInfo(x=gx, score=0, confidence=conf, rank=rank))
                self.overlay_window.update_guides(guides)

    def _on_manual_analyze(self):
        """Tab 키로 수동 분석 트리거 시 호출"""
        with self._analysis_lock:
            if not self._analysis_busy:
                self._manual_trigger = True
                logger.info("수동 분석 요청")

    def _on_bounds_adjusted(self, wall_l: float, wall_r: float,
                            ceiling: float, floor: float):
        """오버레이에서 화살표 키로 경계 조정 시 호출"""
        self._wall_left = wall_l
        self._wall_right = wall_r
        self._ceiling_y = ceiling
        self._floor_y = floor
        # MCTS 엔진에도 즉시 반영
        if self.mcts_engine:
            self.mcts_engine.wall_left = wall_l
            self.mcts_engine.wall_right = wall_r
            self.mcts_engine.ceiling_y = ceiling
            self.mcts_engine.floor_y = floor
        logger.info(f"Bounds adjusted: walls={wall_l:.0f}~{wall_r:.0f}, "
                    f"ceil={ceiling:.0f}, floor={floor:.0f}")
        # 디스크에 저장
        self._save_bounds()

    def _save_bounds(self):
        """경계값을 파일에 저장"""
        import json
        path = os.path.join(os.path.dirname(__file__), 'bounds.json')
        try:
            data = {'wall_left': self._wall_left, 'wall_right': self._wall_right,
                    'ceiling_y': self._ceiling_y, 'floor_y': self._floor_y}
            with open(path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.debug(f"bounds 저장 실패: {e}")

    def _load_bounds(self):
        """저장된 경계값 로드"""
        import json
        path = os.path.join(os.path.dirname(__file__), 'bounds.json')
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            self._wall_left = data.get('wall_left', self._wall_left)
            self._wall_right = data.get('wall_right', self._wall_right)
            self._ceiling_y = data.get('ceiling_y', self._ceiling_y)
            self._floor_y = data.get('floor_y', self._floor_y)
            logger.info(f"Bounds loaded: walls={self._wall_left:.0f}~{self._wall_right:.0f}, "
                        f"ceil={self._ceiling_y:.0f}, floor={self._floor_y:.0f}")
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.debug(f"bounds 로드 실패: {e}")

    def _clamp_x(self, x: float, radius: float) -> float:
        """벽 안쪽으로 x 클램핑"""
        lo = self._wall_left + radius + 5
        hi = self._wall_right - radius - 5
        if lo < hi:
            return max(lo, min(x, hi))
        return (lo + hi) / 2

    def _background_analyze(self, pil_image):
        """백그라운드 스레드에서 Gemini 분석 + GPU MCTS 시뮬레이션 수행"""
        try:
            # --- Stage 1: Gemini API 호출 ---
            self._progress_cb(0.0, "API: Gemini analyzing image...")
            coins, current_coin_type, next_coin_type, game_area = self.gemini_analyzer.analyze_and_extract(pil_image)

            if not current_coin_type:
                logger.warning("현재 동전 미감지")
                self._progress_cb(0.0, "No game detected")
                return

            # --- Stage 2: 게임 영역 업데이트 ---
            wall_left = game_area.get('wall_left_x') or self._wall_left
            wall_right = game_area.get('wall_right_x') or self._wall_right
            ceiling_y = game_area.get('ceiling_y') or self._ceiling_y
            game_score = game_area.get('game_score')
            self._wall_left = wall_left
            self._wall_right = wall_right
            self._ceiling_y = ceiling_y
            if game_score is not None:
                try:
                    self._game_score = int(game_score)
                except (ValueError, TypeError):
                    pass  # 광고로 가려진 경우 무시

            # 오버레이 경계선 업데이트
            if self.overlay_window:
                self.overlay_window.update_bounds(
                    wall_left, wall_right, ceiling_y, self._floor_y)

            self._progress_cb(0.05, f"API done: {len(coins)} coins, score={self._game_score}")
            logger.info(f"동전 {len(coins)}개, 현재: {current_coin_type.display_name}, "
                       f"다음: {next_coin_type.display_name if next_coin_type else '?'}, "
                       f"벽: {wall_left}~{wall_right}, score={self._game_score}")

            # --- Stage 3: 물리 파라미터 자동 보정 ---
            with self._analysis_lock:
                prev_before = self._last_before_state
                prev_drop_coin = self._last_drop_coin
                prev_drop_x = self._last_drop_x

            if self.calibrator and prev_before and prev_drop_coin and coins:
                try:
                    self.calibrator.record(prev_before, prev_drop_coin, prev_drop_x, coins)
                    if len(self.calibrator.history) >= 5 and len(self.calibrator.history) % 5 == 0:
                        self._progress_cb(0.08, "Calibrating physics...")
                        self.calibrator.calibrate(wall_left, wall_right, self._floor_y)
                        calib_path = os.path.join(os.path.dirname(__file__), 'calibration.json')
                        self.calibrator.save(calib_path)
                except Exception as e:
                    logger.warning(f"Calibrator 오류: {e}")

            # --- Stage 4: 최적 위치 계산 ---
            radius = current_coin_type.radius
            self._current_radius = radius  # for progress callback clamping
            guides = []

            if self.mcts_engine:
                # GPU MCTS 딥 서치
                self.mcts_engine.wall_left = wall_left
                self.mcts_engine.wall_right = wall_right
                self.mcts_engine.ceiling_y = ceiling_y
                self.mcts_engine.floor_y = self._floor_y
                best_x, best_score, details = self.mcts_engine.search(
                    coins, current_coin_type, next_coin_type,
                    progress_callback=self._progress_cb
                )

                # 여러 후보를 GuideInfo로 변환
                candidates = details.get('candidates', [])
                for c in candidates:
                    gx = self._clamp_x(c['x'], radius)
                    guides.append(GuideInfo(
                        x=gx, score=c['score'],
                        confidence=c['confidence'], rank=c['rank']
                    ))

                if not guides:
                    gx = self._clamp_x(best_x, radius)
                    guides.append(GuideInfo(x=gx, score=best_score, confidence=1.0, rank=0))

                depth = details.get('depth_reached', 0)
                iters = details.get('iterations', 0)
                info_str = f"MCTS d={depth} i={iters} | score={self._game_score}"
            else:
                # 폴백: 기본 optimizer
                self._progress_cb(0.1, "Computing (basic optimizer)...")
                best_x, best_score, details = self.position_optimizer.find_optimal_position(
                    coins, current_coin_type,
                    next_coin_type=next_coin_type,
                    wall_left_x=wall_left, wall_right_x=wall_right
                )
                gx = self._clamp_x(best_x, radius)
                guides.append(GuideInfo(x=gx, score=best_score, confidence=1.0, rank=0))
                info_str = f"basic | score={self._game_score}"

            best_x = guides[0].x if guides else (wall_left + wall_right) / 2

            # --- Stage 5: 결과 저장 + 오버레이 업데이트 ---
            with self._analysis_lock:
                self._last_before_state = coins
                self._last_drop_coin = current_coin_type
                self._last_drop_x = best_x
                try:
                    predicted, _ = self.physics_simulator.simulate_drop(coins, current_coin_type, best_x)
                    self._last_predicted = predicted
                except Exception:
                    self._last_predicted = None
                self._last_result = (guides, best_score, info_str)

        except Exception as e:
            logger.error(f"백그라운드 분석 오류: {e}", exc_info=True)
            self._progress_cb(0.0, f"Error: {e}")
        finally:
            with self._analysis_lock:
                self._analysis_busy = False

    def _wait_for_motion_start(self, max_wait_s: float = 30.0):
        """추천 후 사용자가 드롭해서 화면이 다시 '움직이기 시작'할 때까지 대기."""
        if not self.screen_capture or not self.state_detector or not self.overlay_window:
            return

        start = time.time()
        sleep_s = max(0.01, 1.0 / max(1, config.CAPTURE_FPS))

        while time.time() - start < max_wait_s:
            if not self.overlay_window.is_open():
                return

            frame = self.screen_capture.capture()
            if frame is None:
                time.sleep(sleep_s)
                continue

            self.state_detector.add_frame(frame)
            if not self.state_detector.is_stable():
                return

            self.overlay_window.update()
            time.sleep(sleep_s)
    
    def cleanup(self):
        """리소스 정리"""
        logger.info("리소스 정리 중...")
        
        if self.overlay_window:
            self.overlay_window.close()
        
        self.running = False
        logger.info("프로그램 종료")


def main():
    """메인 함수"""
    print("\n"
          "============================================================\n"
          "                                                            \n"
          "          [Coin Game Assistant]                              \n"
          "          -- Dong-jeon Game Real-time Guide --               \n"
          "                                                            \n"
          "   MuMu Player eseo dong-jeon game-eul silhaeng han hu      \n"
          "   i program-eul sijakhaseyo.                                \n"
          "   Game hwamyeon wie choejeog-ui naghha wichi-ga             \n"
          "   pyosidoemnida.                                           \n"
          "                                                            \n"
          "   Jong-ryo: Ctrl+C or overlay chang dadgi                  \n"
          "                                                            \n"
          "============================================================\n"
          )
    
    # CoinGameAssistant 인스턴스 생성
    assistant = CoinGameAssistant()
    
    # 초기화
    if not assistant.initialize():
        print("\n[FAIL] Initialization failed. Check logs.")
        return 1
    
    print("\n[OK] Initialization complete! Starting game analysis...\n")
    
    # 메인 루프 실행
    assistant.run()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
