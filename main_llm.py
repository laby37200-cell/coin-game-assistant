"""
동전게임 LLM-Only 공략 가이드 - 메인 프로그램

물리 시뮬레이터 없이 Gemini Flash 3.0 Preview가 직접 게임 화면을 보고
최적의 낙하 위치를 판단합니다. 이전/현재 상태를 비교하여 자체 피드백합니다.
"""

import os
import sys
import time
import logging
import logging.handlers
import threading
import json
import gc
import traceback
import tkinter as tk
from typing import Optional

# Windows 콘솔 UTF-8 출력 보장
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# 프로젝트 루트를 PYTHONPATH에 추가
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _APP_DIR)

import config
from vision.screen_capture import ScreenCapture
from ui.overlay import OverlayWindow, GuideInfo
from utils.state_detector import StateDetector

from solver.llm_advisor import LLMAdvisor


# ── 로깅 설정: 콘솔 + 파일(crash_log.txt) ──
_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(
    level=getattr(logging, config.DEBUG_LOG_LEVEL),
    format=_LOG_FORMAT
)
# 크래시 로그 파일 (최대 2MB × 3개 로테이션)
_crash_log_path = os.path.join(_APP_DIR, 'crash_log.txt')
try:
    _file_handler = logging.handlers.RotatingFileHandler(
        _crash_log_path, maxBytes=2*1024*1024, backupCount=3,
        encoding='utf-8')
    _file_handler.setLevel(logging.DEBUG)
    _file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logging.getLogger().addHandler(_file_handler)
except Exception:
    pass

logger = logging.getLogger(__name__)


# ── 전역 예외 훅: 잡히지 않은 예외를 파일에 기록 ──
def _global_exception_hook(exc_type, exc_value, exc_tb):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return
    logger.critical("잡히지 않은 예외 (메인 스레드):",
                    exc_info=(exc_type, exc_value, exc_tb))

def _thread_exception_hook(args):
    logger.critical(f"잡히지 않은 예외 (스레드 {args.thread.name}):",
                    exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

sys.excepthook = _global_exception_hook
if hasattr(threading, 'excepthook'):
    threading.excepthook = _thread_exception_hook


class LLMGameAssistant:
    """LLM-Only 동전게임 공략 가이드"""

    def __init__(self):
        logger.info("=== LLM-Only 동전게임 가이드 시작 ===")

        self.screen_capture: Optional[ScreenCapture] = None
        self.overlay_window: Optional[OverlayWindow] = None
        self.state_detector: Optional[StateDetector] = None
        self.llm_advisor: Optional[LLMAdvisor] = None

        self.game_width = config.GAME_WIDTH
        self.game_height = config.GAME_HEIGHT
        self.running = False
        self._game_score = 0

        # 자동 터치 모드
        self._auto_touch = False
        self._touch_y = 350  # 터치할 y좌표 (동전 드롭 영역 — 천장 부근)

        # 안정성: 분석 카운터 & 워치독
        self._analysis_count = 0
        self._last_analysis_time = 0.0
        self._consecutive_errors = 0
        self._MAX_CONSECUTIVE_ERRORS = 5

        # 경계값
        self._wall_left = 78.0
        self._wall_right = 468.0
        self._ceiling_y = 380.0
        self._floor_y = 930.0
        self._load_bounds()

    # ------------------------------------------------------------------ #
    # 초기화
    # ------------------------------------------------------------------ #
    def initialize(self) -> bool:
        try:
            # 1. 화면 캡처
            logger.info("화면 캡처 초기화 중...")
            self.screen_capture = ScreenCapture(config.WINDOW_TITLE_PATTERN)
            if not self.screen_capture.find_window():
                logger.error("MuMu Player 창을 찾을 수 없습니다.")
                return False

            dimensions = self.screen_capture.get_game_dimensions()
            if dimensions:
                self.game_width, self.game_height = dimensions
                logger.info(f"게임 영역: {self.game_width}x{self.game_height}")

            # 2. LLM 어드바이저
            logger.info("LLM 어드바이저 초기화 중...")
            self.llm_advisor = LLMAdvisor(
                api_key=config.GEMINI_API_KEY,
                model_name="gemini-3-flash-preview",
            )
            self.llm_advisor.wall_left = self._wall_left
            self.llm_advisor.wall_right = self._wall_right
            self.llm_advisor.ceiling_y = self._ceiling_y
            self.llm_advisor.floor_y = self._floor_y

            # 3. 오버레이
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
            # 경계값을 창 생성 전에 설정 (컨트롤 패널이 올바른 초기값 표시)
            self.overlay_window._wall_left = self._wall_left
            self.overlay_window._wall_right = self._wall_right
            self.overlay_window._ceiling_y = self._ceiling_y
            self.overlay_window._floor_y = self._floor_y
            self.overlay_window.create_window()
            self.overlay_window.update()
            self.overlay_window.set_bounds_callback(self._on_bounds_adjusted)
            self.overlay_window.set_analyze_callback(self._on_manual_analyze)
            self.overlay_window.set_chat_callback(self._on_chat_message)
            self.overlay_window.set_auto_touch_callback(self._on_toggle_auto_touch)

            # 4. 상태 감지기
            logger.info("상태 감지기 초기화 중...")
            self.state_detector = StateDetector(
                check_frames=config.STABILITY_CHECK_FRAMES,
                pixel_threshold=config.STABILITY_PIXEL_THRESHOLD,
                wait_time=config.STABILITY_WAIT_TIME
            )

            logger.info("LLM-Only 초기화 완료")
            return True

        except Exception as e:
            logger.error(f"초기화 실패: {e}", exc_info=True)
            return False

    # ------------------------------------------------------------------ #
    # 메인 루프
    # ------------------------------------------------------------------ #
    def run(self):
        self.running = True
        logger.info("메인 루프 시작")

        self._analysis_lock = threading.Lock()
        self._analysis_busy = False
        self._manual_trigger = False
        self._last_result = None
        self._gc_counter = 0

        self.overlay_window.show_message(
            "LLM-Only 동전게임 가이드\n컨트롤 패널에서 조작하세요",
            duration=5000)

        try:
            while self.running:
                # ── Tkinter 이벤트 처리 ──
                try:
                    self.overlay_window.update()
                except tk.TclError:
                    logger.warning("오버레이 TclError — 종료")
                    break
                except Exception as e:
                    logger.error(f"오버레이 update 오류: {e}", exc_info=True)
                    break

                if not self.overlay_window.is_open():
                    logger.info("오버레이 창이 닫혔습니다.")
                    break

                # ── 결과 수신 (항상 처리) ──
                try:
                    with self._analysis_lock:
                        if self._last_result is not None:
                            guides, info_str, reason = self._last_result
                            self._last_result = None
                            if isinstance(guides, list) and guides:
                                self.overlay_window.update_guides(guides)
                                self.overlay_window.update_status(info_str)
                                if reason:
                                    self.overlay_window.show_message(reason, duration=8000)
                except Exception as e:
                    logger.debug(f"결과 수신 오류: {e}")

                # ── 화면 캡처 ──
                try:
                    screenshot = self.screen_capture.capture()
                except Exception as e:
                    logger.warning(f"캡처 오류: {e}")
                    screenshot = None
                if screenshot is None:
                    time.sleep(0.5)
                    continue

                # ── 안정성 체크 ──
                self.state_detector.add_frame(screenshot)
                del screenshot  # 즉시 해제
                is_stable = self.state_detector.is_stable()

                # ── 워치독: 분석이 120초 이상 걸리면 강제 해제 ──
                with self._analysis_lock:
                    busy = self._analysis_busy
                    if busy and self._last_analysis_time > 0:
                        stuck_sec = time.time() - self._last_analysis_time
                        if stuck_sec > 120:
                            logger.warning(f"분석 워치독: {stuck_sec:.0f}초 경과 — 강제 해제")
                            self._analysis_busy = False
                            busy = False
                    trigger = self._manual_trigger
                    self._manual_trigger = False

                # ── 연속 에러 시 쿨다운 ──
                if self._consecutive_errors >= self._MAX_CONSECUTIVE_ERRORS:
                    if self._auto_touch:
                        logger.warning(f"연속 에러 {self._consecutive_errors}회 — 30초 쿨다운")
                        self.overlay_window.update_status(
                            f"⚠ 연속 에러 {self._consecutive_errors}회 — 30초 대기")
                        for _ in range(300):  # 30초, 0.1초 단위
                            if not self.running:
                                break
                            time.sleep(0.1)
                        self._consecutive_errors = 0
                        continue

                # ── 분석 시작 판단 ──
                auto_trigger = self._auto_touch and is_stable and (not busy)
                should_analyze = (not busy) and (trigger or auto_trigger)

                if should_analyze:
                    try:
                        pil_image = self.screen_capture.capture_pil()
                    except Exception:
                        pil_image = None
                    if pil_image is not None:
                        with self._analysis_lock:
                            self._analysis_busy = True
                            self._last_analysis_time = time.time()
                        mode_label = "[자동]" if self._auto_touch else "[수동]"
                        self.overlay_window.update_status(f"{mode_label} LLM 분석 중...")
                        self.overlay_window.update_progress(0.0)
                        t = threading.Thread(
                            target=self._safe_background_analyze,
                            args=(pil_image,), daemon=True,
                            name="LLM-Analysis")
                        t.start()

                # ── 주기적 GC (50회마다) ──
                self._gc_counter += 1
                if self._gc_counter >= 50:
                    self._gc_counter = 0
                    gc.collect()

                # ── 메인 루프 속도 제한 (최소 50ms) ──
                time.sleep(max(0.05, 1.0 / max(1, config.CAPTURE_FPS)))

        except KeyboardInterrupt:
            logger.info("사용자에 의해 중단됨")
        except Exception as e:
            logger.critical(f"메인 루프 치명적 오류: {e}", exc_info=True)
        finally:
            self.cleanup()

    # ------------------------------------------------------------------ #
    # 백그라운드 분석
    # ------------------------------------------------------------------ #
    def _safe_background_analyze(self, pil_image):
        """백그라운드 스레드 래퍼 — 모든 예외를 잡아서 로깅"""
        try:
            self._background_analyze(pil_image)
        except Exception as e:
            logger.error(f"분석 스레드 치명적 오류: {e}", exc_info=True)
            self._consecutive_errors += 1
        finally:
            with self._analysis_lock:
                self._analysis_busy = False
            # PIL 이미지 명시적 해제
            try:
                pil_image.close()
            except Exception:
                pass
            del pil_image

    def _background_analyze(self, pil_image):
        """백그라운드 스레드에서 LLM 분석 수행"""
        start_time = time.time()
        result = self.llm_advisor.analyze(
            pil_image,
            progress_callback=self._progress_cb
        )
        elapsed = time.time() - start_time

        if not result:
            self._progress_cb(0.0, "LLM 분석 실패")
            self._consecutive_errors += 1
            return

        self._consecutive_errors = 0
        self._analysis_count += 1

        # 결과 파싱
        drop_x = result.get("drop_x")
        confidence = result.get("confidence", 0.5)
        reason = result.get("reason", "")
        strategy = result.get("strategy", "")
        risk = result.get("risk_level", "safe")
        alt_x = result.get("alternative_x")
        alt_reason = result.get("alternative_reason", "")
        game_score = result.get("game_score")

        if game_score is not None:
            try:
                self._game_score = int(game_score)
            except (ValueError, TypeError):
                pass

        # 게임오버 위험도 체크 (highest_coin_y vs ceiling)
        highest_y = result.get("highest_coin_y")
        if highest_y is not None:
            try:
                hy = float(highest_y)
                margin = hy - self._ceiling_y
                if margin < 50:
                    risk = "danger"
                    result["risk_level"] = "danger"
                    logger.warning(f"🚨 게임오버 위험! highest_y={hy:.0f}, ceiling={self._ceiling_y:.0f}, margin={margin:.0f}")
                elif margin < 100:
                    if risk == "safe":
                        risk = "warning"
                        result["risk_level"] = "warning"
            except (ValueError, TypeError):
                pass

        # 가이드 라인 생성
        guides = []
        if drop_x is not None:
            # 전략명을 간략 설명으로 사용
            main_desc = strategy[:12] if strategy else ""
            guides.append(GuideInfo(
                x=float(drop_x), score=self._game_score,
                confidence=float(confidence), rank=0,
                desc=main_desc))

        if alt_x is not None and alt_x != drop_x:
            alt_desc = alt_reason[:12] if alt_reason else "대안"
            guides.append(GuideInfo(
                x=float(alt_x), score=0,
                confidence=max(0.1, float(confidence) - 0.3), rank=1,
                desc=alt_desc))

        # 위험도에 따른 색상 힌트
        risk_emoji = {"safe": "✅", "warning": "⚠️", "danger": "🚨"}.get(risk, "")

        # 히스토리 요약
        hist = self.llm_advisor.get_history_summary()

        info_str = (f"LLM {risk_emoji} | {strategy} | "
                    f"score={self._game_score} | {hist}")

        # 이유 텍스트 (오버레이 메시지용)
        display_reason = f"{risk_emoji} {reason}"
        if alt_reason:
            display_reason += f"\n차선: {alt_reason}"

        with self._analysis_lock:
            self._last_result = (guides, info_str, display_reason)

        # 분석 로그 창에 결과 전달
        if self.overlay_window:
            try:
                self.overlay_window.add_analysis_result(result, elapsed)
            except Exception:
                pass

        # ── 자동 터치 모드: 분석 완료 후 자동 클릭 ──
        if self._auto_touch and drop_x is not None and self.running:
            self._execute_touch(float(drop_x))

        logger.info(f"분석 #{self._analysis_count} 완료 ({elapsed:.1f}s)")

    # ------------------------------------------------------------------ #
    # 콜백
    # ------------------------------------------------------------------ #
    def _progress_cb(self, progress: float, status: str):
        if not self.overlay_window or not self.running:
            return
        try:
            self.overlay_window.update_progress(progress)
            self.overlay_window.update_status(status)
        except Exception:
            pass

    def _execute_touch(self, drop_x: float):
        """자동 터치: drop_x 위치에 클릭 전송 후 안정화 대기"""
        if not self.running or not self.screen_capture:
            return
        try:
            time.sleep(0.3)
            if not self.running:
                return
            ok = self.screen_capture.click_at(int(drop_x), self._touch_y)
            if ok:
                logger.info(f"자동 터치: x={int(drop_x)}, y={self._touch_y}")
                self._progress_cb(1.0, f"[자동] 터치 x={int(drop_x)}")
                if self.state_detector:
                    self.state_detector.reset()
                # 터치 후 안정화 대기 (2초, 0.1초 단위 — 종료 체크)
                for _ in range(20):
                    if not self.running:
                        return
                    time.sleep(0.1)
            else:
                logger.warning("자동 터치 실패")
                self._progress_cb(0.0, "[자동] 터치 실패")
        except Exception as e:
            logger.error(f"자동 터치 오류: {e}", exc_info=True)

    def _on_toggle_auto_touch(self):
        """자동 터치 모드 토글"""
        self._auto_touch = not self._auto_touch
        state = "ON" if self._auto_touch else "OFF"
        logger.info(f"자동 터치 모드: {state}")
        if self.overlay_window:
            self.overlay_window.show_message(
                f"🤖 자동 터치 모드: {state}", duration=3000)
        # 자동 모드 켜질 때 즉시 첫 분석 트리거
        if self._auto_touch:
            with self._analysis_lock:
                if not self._analysis_busy:
                    self._manual_trigger = True

    def _on_manual_analyze(self):
        """Tab 키로 수동 분석 트리거"""
        with self._analysis_lock:
            if not self._analysis_busy:
                self._manual_trigger = True
                logger.info("수동 분석 요청")

    def _on_chat_message(self, user_message: str) -> str:
        """사용자 채팅 메시지를 LLM에게 전달하고 응답 반환"""
        if not self.llm_advisor:
            return "LLM 어드바이저가 초기화되지 않았습니다."
        try:
            # 최신 게임 화면 캡처하여 함께 전달
            pil_image = None
            if self.screen_capture:
                pil_image = self.screen_capture.capture_pil()
            return self.llm_advisor.chat(user_message, image=pil_image)
        except Exception as e:
            logger.error(f"채팅 오류: {e}", exc_info=True)
            return f"오류: {e}"

    def _on_bounds_adjusted(self, wall_l, wall_r, ceiling, floor):
        """경계 조정 콜백"""
        self._wall_left = wall_l
        self._wall_right = wall_r
        self._ceiling_y = ceiling
        self._floor_y = floor
        if self.llm_advisor:
            self.llm_advisor.wall_left = wall_l
            self.llm_advisor.wall_right = wall_r
            self.llm_advisor.ceiling_y = ceiling
            self.llm_advisor.floor_y = floor
        logger.info(f"Bounds: walls={wall_l:.0f}~{wall_r:.0f}, "
                    f"ceil={ceiling:.0f}, floor={floor:.0f}")
        self._save_bounds()

    # ------------------------------------------------------------------ #
    # 경계값 저장/로드
    # ------------------------------------------------------------------ #
    def _save_bounds(self):
        path = os.path.join(os.path.dirname(__file__), 'bounds.json')
        try:
            data = {'wall_left': self._wall_left, 'wall_right': self._wall_right,
                    'ceiling_y': self._ceiling_y, 'floor_y': self._floor_y}
            with open(path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.debug(f"bounds 저장 실패: {e}")

    def _load_bounds(self):
        path = os.path.join(os.path.dirname(__file__), 'bounds.json')
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            self._wall_left = data.get('wall_left', self._wall_left)
            self._wall_right = data.get('wall_right', self._wall_right)
            self._ceiling_y = data.get('ceiling_y', self._ceiling_y)
            self._floor_y = data.get('floor_y', self._floor_y)
            logger.info(f"Bounds loaded: walls={self._wall_left:.0f}~{self._wall_right:.0f}")
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.debug(f"bounds 로드 실패: {e}")

    def cleanup(self):
        logger.info("리소스 정리 중...")
        self.running = False
        try:
            if self.overlay_window:
                self.overlay_window.close()
        except Exception as e:
            logger.debug(f"오버레이 종료 오류: {e}")
        # 메모리 정리
        if self.llm_advisor:
            try:
                self.llm_advisor._last_image = None
                self.llm_advisor._last_board_state = None
            except Exception:
                pass
        gc.collect()
        logger.info(f"프로그램 종료 (총 분석 {self._analysis_count}회)")


def main():
    print("\n"
          "============================================================\n"
          "                                                            \n"
          "          [Coin Game Assistant - LLM Only]                   \n"
          "          물리엔진 없이 AI가 직접 판단합니다                  \n"
          "                                                            \n"
          "   조작법:                                                   \n"
          "     왼쪽 컨트롤 패널의 버튼을 사용하세요              \n"
          "     [▶ 분석 시작]  AI 분석 1회 실행                    \n"
          "     [📐 경계 편집]  경계선 조정 모드                     \n"
          "     [✕ 종료]       프로그램 종료                       \n"
          "                                                            \n"
          "============================================================\n"
          )

    assistant = LLMGameAssistant()

    if not assistant.initialize():
        print("\n[FAIL] 초기화 실패. 로그를 확인하세요.")
        return 1

    print("\n[OK] 초기화 완료! Tab을 눌러 분석을 시작하세요.\n")

    assistant.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
