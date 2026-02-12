"""
오버레이 UI

tkinter를 사용하여 게임 화면 위에 투명한 가이드 라인을 표시합니다.
- 여러 개의 가이드 라인을 확률별 색상으로 표시
- 추론 진행도 표시
- 스레드 안전한 업데이트 (큐 기반)
"""

import tkinter as tk
import logging
import queue
import ctypes
from typing import Optional, Tuple, List


logger = logging.getLogger(__name__)

# 투명 배경으로 사용할 색상 키
_TRANSPARENT_KEY = '#010101'


class GuideInfo:
    """하나의 가이드 라인 정보"""
    __slots__ = ('x', 'score', 'confidence', 'rank', 'desc')

    def __init__(self, x: float, score: float, confidence: float, rank: int,
                 desc: str = ''):
        self.x = x
        self.score = score
        self.confidence = confidence  # 0.0 ~ 1.0
        self.rank = rank              # 0 = best
        self.desc = desc              # 간략한 설명


def confidence_color(confidence: float) -> str:
    """확률/신뢰도에 따른 색상 반환"""
    if confidence >= 0.9:
        return '#00FF00'   # 초록 — 최적
    elif confidence >= 0.75:
        return '#AAFF00'   # 연두
    elif confidence >= 0.6:
        return '#FFFF00'   # 노랑
    elif confidence >= 0.4:
        return '#FFAA00'   # 주황
    else:
        return '#FF4444'   # 빨강 — 위험


class OverlayWindow:
    """투명 오버레이 윈도우 클래스 (스레드 안전)"""

    # 큐 메시지 타입
    _MSG_GUIDES = 'guides'
    _MSG_STATUS = 'status'
    _MSG_PROGRESS = 'progress'
    _MSG_MESSAGE = 'message'
    _MSG_BOUNDS = 'bounds'

    def __init__(
        self,
        window_x: int,
        window_y: int,
        window_width: int,
        window_height: int,
        opacity: float = 0.85,
        line_color: str = "#00FF00",
        line_width: int = 3,
        text_color: str = "#FFFFFF",
        text_size: int = 14
    ):
        self.window_x = window_x
        self.window_y = window_y
        self.window_width = window_width
        self.window_height = window_height
        self.opacity = opacity
        self.line_color = line_color
        self.line_width = line_width
        self.text_color = text_color
        self.text_size = text_size

        self.root = None
        self.canvas = None

        # 스레드 안전 큐 — 백그라운드 스레드에서 UI 업데이트 요청 (크기 제한)
        self._queue = queue.Queue(maxsize=200)

        # 현재 표시 상태
        self._guides: List[GuideInfo] = []
        self._status_text = ''
        self._progress = 0.0  # 0.0 ~ 1.0
        self._polling = False
        self._after_id = None

        # 벽/천장/바닥 경계선
        self._wall_left = 0.0
        self._wall_right = float(window_width)
        self._ceiling_y = 0.0
        self._floor_y = float(window_height)
        self._bounds_visible = True

        # 콜백
        self._bounds_callback = None  # callable(wall_l, wall_r, ceil, floor)
        self._analyze_callback = None   # callable() — 수동 분석 트리거
        self._chat_callback = None      # callable(msg) → str — LLM 대화
        self._auto_touch_callback = None  # callable() — 자동 터치 모드 토글
        self._control_panel = None       # 컨트롤 패널 참조
        self._hwnd = None               # 오버레이 HWND

        self._bounds_editing = False  # 하위 호환

        logger.info(f"OverlayWindow init: {window_width}x{window_height} at ({window_x},{window_y})")

    # ------------------------------------------------------------------ #
    # Window lifecycle
    # ------------------------------------------------------------------ #
    def create_window(self):
        """오버레이 윈도우 생성"""
        self.root = tk.Tk()
        self.root.title("Coin Game Assistant")
        # 테두리/타이틀바 제거 — 캔버스가 정확히 게임 창 위에 겹침
        self.root.overrideredirect(True)
        self.root.geometry(f"{self.window_width}x{self.window_height}+{self.window_x}+{self.window_y}")
        self.root.attributes('-topmost', True)

        # 투명 배경 (Windows)
        try:
            self.root.attributes('-transparentcolor', _TRANSPARENT_KEY)
            self.root.attributes('-alpha', self.opacity)
        except Exception:
            logger.warning("투명 배경 설정 실패")

        # Windows: 클릭 투과 — 투명 영역은 클릭이 게임으로 전달됨
        try:
            self._hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
            style = ctypes.windll.user32.GetWindowLongW(self._hwnd, -20)  # GWL_EXSTYLE
            # WS_EX_LAYERED 만 설정 (WS_EX_TRANSPARENT 는 렌더링 문제 유발 가능)
            style |= 0x80000  # WS_EX_LAYERED
            ctypes.windll.user32.SetWindowLongW(self._hwnd, -20, style)
        except Exception:
            logger.warning("레이어드 윈도우 설정 실패")

        self.canvas = tk.Canvas(
            self.root,
            width=self.window_width,
            height=self.window_height,
            bg=_TRANSPARENT_KEY,
            highlightthickness=0
        )
        self.canvas.pack()

        self.root.protocol("WM_DELETE_WINDOW", self.close)

        # ESC / Ctrl+Q 로 종료
        self.root.bind('<Escape>', lambda e: self.close())
        self.root.bind('<Control-q>', lambda e: self.close())

        self.root.bind('b', lambda e: self._toggle_bounds())

        # 컨트롤 패널 생성 (버튼 UI)
        from ui.control_panel import ControlPanel
        self._control_panel = ControlPanel(self.root, self)

        # 큐 폴링 시작 (50ms 간격)
        self._polling = True
        self._poll_queue()

        logger.info("오버레이 윈도우 생성 완료")

    def _poll_queue(self):
        """메인 스레드에서 큐를 폴링하여 UI 업데이트 (스레드 안전)"""
        if not self.root or not self._polling:
            return
        try:
            while not self._queue.empty():
                msg_type, data = self._queue.get_nowait()
                if msg_type == self._MSG_GUIDES:
                    self._draw_guides(data)
                elif msg_type == self._MSG_STATUS:
                    self._status_text = data
                    self._redraw()
                elif msg_type == self._MSG_PROGRESS:
                    self._progress = data
                    self._redraw()
                elif msg_type == self._MSG_MESSAGE:
                    text, duration = data
                    self._draw_message(text, duration)
                elif msg_type == self._MSG_BOUNDS:
                    wl, wr, cy, fy = data
                    self._wall_left = wl
                    self._wall_right = wr
                    self._ceiling_y = cy
                    self._floor_y = fy
                    self._redraw()
                    # 컨트롤 패널 입력 필드 동기화
                    if self._control_panel:
                        self._control_panel.update_bounds_display(wl, wr, cy, fy)
                elif msg_type == '_analysis_result':
                    result, elapsed = data
                    if self._control_panel:
                        self._control_panel.add_analysis_result(result, elapsed)
        except Exception:
            pass
        if self.root and self._polling:
            try:
                self._after_id = self.root.after(50, self._poll_queue)
            except (tk.TclError, RuntimeError):
                pass

    # ------------------------------------------------------------------ #
    # Public API (thread-safe — can be called from any thread)
    # ------------------------------------------------------------------ #
    def update_guide(self, guide_x: float, score: float):
        """단일 가이드 업데이트 (하위 호환)"""
        self.update_guides([GuideInfo(x=guide_x, score=score, confidence=1.0, rank=0)])

    def _safe_put(self, msg):
        """큐에 비차단 삽입 — 큐가 가득 차면 드롭"""
        try:
            self._queue.put_nowait(msg)
        except queue.Full:
            pass  # 오래된 메시지 드롭 (UI 지연보다 안정성 우선)

    def update_guides(self, guides: List[GuideInfo]):
        """여러 가이드 라인 업데이트 (스레드 안전)"""
        self._safe_put((self._MSG_GUIDES, guides))

    def update_status(self, text: str):
        """상태 텍스트 업데이트 (스레드 안전)"""
        self._safe_put((self._MSG_STATUS, text))

    def update_progress(self, progress: float):
        """진행도 업데이트 0.0~1.0 (스레드 안전)"""
        self._safe_put((self._MSG_PROGRESS, max(0.0, min(1.0, progress))))

    def show_message(self, message: str, duration: int = 3000):
        """일시적 메시지 표시 (스레드 안전)"""
        self._safe_put((self._MSG_MESSAGE, (message, duration)))

    def clear_guide(self):
        """가이드 제거"""
        self._guides = []
        self._safe_put((self._MSG_GUIDES, []))

    def update_bounds(self, wall_left: float, wall_right: float,
                      ceiling_y: float, floor_y: float):
        """벽/천장/바닥 경계 업데이트 (스레드 안전)"""
        self._safe_put((self._MSG_BOUNDS, (wall_left, wall_right, ceiling_y, floor_y)))

    def set_bounds_callback(self, callback):
        """경계 조정 시 호출될 콜백 설정: callback(wall_l, wall_r, ceil, floor)"""
        self._bounds_callback = callback

    def set_analyze_callback(self, callback):
        """수동 분석 트리거 콜백 설정: callback()"""
        self._analyze_callback = callback

    def set_chat_callback(self, callback):
        """LLM 대화 콜백 설정: callback(user_msg) → str"""
        self._chat_callback = callback

    def set_auto_touch_callback(self, callback):
        """자동 터치 모드 토글 콜백 설정: callback()"""
        self._auto_touch_callback = callback

    def add_analysis_result(self, result: dict, elapsed: float = 0.0):
        """분석 결과를 분석 로그 창에 전달 (스레드 안전)"""
        if self._control_panel:
            self._safe_put(('_analysis_result', (result, elapsed)))

    @property
    def auto_mode(self) -> bool:
        return False

    def _notify_bounds(self):
        """콜백으로 main에 경계값 변경 알림 (저장 포함)"""
        if self._bounds_callback:
            self._bounds_callback(
                self._wall_left, self._wall_right,
                self._ceiling_y, self._floor_y)

    def _toggle_bounds(self):
        """'b' 키로 경계선 표시/숨기기 토글"""
        self._bounds_visible = not self._bounds_visible
        self._redraw()

    # ------------------------------------------------------------------ #
    # Drawing (메인 스레드에서만 호출)
    # ------------------------------------------------------------------ #
    def _redraw(self):
        """전체 캔버스 다시 그리기"""
        if not self.canvas:
            return
        self.canvas.delete('all')
        self._draw_bounds()
        self._draw_guide_lines()
        self._draw_status_bar()

    def _draw_guides(self, guides: List[GuideInfo]):
        """가이드 라인들 그리기"""
        self._guides = guides
        self._redraw()

    def _draw_bounds(self):
        """벽/천장/바닥 경계선 그리기"""
        if not self.canvas or not self._bounds_visible:
            return

        wl = self._wall_left
        wr = self._wall_right
        cy = self._ceiling_y
        fy = self._floor_y

        line_defs = [
            (True,  wl, '#00AAAA', f'L:{wl:.0f}'),
            (True,  wr, '#00AAAA', f'R:{wr:.0f}'),
            (False, cy, '#FF4444', f'Ceil:{cy:.0f}'),
            (False, fy, '#44AA44', f'Floor:{fy:.0f}'),
        ]

        for is_vert, pos, color, label in line_defs:
            if is_vert:
                self.canvas.create_line(
                    pos, 0, pos, self.window_height,
                    fill=color, width=1, dash=(4, 8))
                self.canvas.create_text(
                    pos + 3, 15, text=label, fill=color,
                    font=('Consolas', 7), anchor='w')
            else:
                self.canvas.create_line(
                    0, pos, self.window_width, pos,
                    fill=color, width=1, dash=(4, 8))
                self.canvas.create_text(
                    self.window_width - 5, pos - 8, text=label,
                    fill=color, font=('Consolas', 7), anchor='e')

    def _draw_guide_lines(self):
        """현재 저장된 가이드 라인들을 캔버스에 그리기"""
        if not self.canvas or not self._guides:
            return

        for g in reversed(self._guides):  # 낮은 순위부터 그려서 최고가 위에
            color = confidence_color(g.confidence)
            width = self.line_width + (2 if g.rank == 0 else 0)

            # 경계선 범위 내로 클램핑
            wl = self._wall_left
            wr = self._wall_right
            cy = self._ceiling_y
            fy = self._floor_y
            gx = max(wl + 5, min(g.x, wr - 5))
            top_y = max(0, cy - 30)

            # 수직 가이드 라인 (천장~바닥)
            self.canvas.create_line(
                gx, top_y, gx, fy,
                fill=color, width=width, dash=(6, 3) if g.rank > 0 else ()
            )

            # 라인 상단에 삼각형 마커
            self.canvas.create_polygon(
                gx - 8, top_y, gx + 8, top_y, gx, top_y + 15,
                fill=color, outline=color
            )

            # 확률 텍스트 + 간략 설명
            pct = int(g.confidence * 100)
            # 설명이 있으면 사용, 없으면 신뢰도 기반 자동 생성
            if g.desc:
                desc = g.desc
            elif pct >= 90:
                desc = "최적"
            elif pct >= 75:
                desc = "좋음"
            elif pct >= 60:
                desc = "보통"
            elif pct >= 40:
                desc = "주의"
            else:
                desc = "위험"
            label = f"{pct}% {desc}"
            tx = gx
            if tx < 60:
                tx = 60
            elif tx > self.window_width - 60:
                tx = self.window_width - 60

            self.canvas.create_text(
                tx, max(10, top_y - 10),
                text=label,
                fill=color,
                font=('맑은 고딕', 10, 'bold'),
            )

        # 최고 가이드 정보 텍스트
        best = self._guides[0] if self._guides else None
        if best:
            best_desc = best.desc if best.desc else "DROP HERE"
            info = f"{best_desc}  (x={best.x:.0f})"
            tx = max(5, min(best.x, self.window_width - 5))
            if tx < 90:
                tx = 90
            elif tx > self.window_width - 90:
                tx = self.window_width - 90

            # 텍스트 배경 박스
            box_w = max(80, len(info) * 7)
            self.canvas.create_rectangle(
                tx - box_w // 2, 38, tx + box_w // 2, 60,
                fill='#222222', outline=confidence_color(best.confidence), width=1
            )
            self.canvas.create_text(
                tx, 49,
                text=info,
                fill='#FFFFFF',
                font=('맑은 고딕', 9, 'bold'),
            )

    def _draw_status_bar(self):
        """하단 상태 바 + 진행도 표시"""
        if not self.canvas:
            return

        bar_h = 28
        bar_y = self.window_height - bar_h

        # 상태 바 배경
        self.canvas.create_rectangle(
            0, bar_y, self.window_width, self.window_height,
            fill='#1a1a1a', outline='#333333'
        )

        # 진행도 바
        if self._progress > 0:
            prog_w = int(self.window_width * self._progress)
            color = '#00CC66' if self._progress >= 1.0 else '#3399FF'
            self.canvas.create_rectangle(
                0, bar_y, prog_w, bar_y + 4,
                fill=color, outline=''
            )

        # 상태 텍스트
        status = self._status_text or 'Ready'
        self.canvas.create_text(
            10, bar_y + 16,
            text=status,
            fill='#CCCCCC',
            font=('Consolas', 9),
            anchor='w'
        )

        # 진행도 퍼센트
        if self._progress > 0:
            self.canvas.create_text(
                self.window_width - 10, bar_y + 16,
                text=f"{int(self._progress * 100)}%",
                fill='#AAAAAA',
                font=('Consolas', 9),
                anchor='e'
            )

    def _draw_message(self, message: str, duration: int):
        """일시적 메시지 표시"""
        if not self.canvas:
            return

        # 반투명 배경 박스 (한글 2~3줄에 맞게 크기 조정)
        cx = self.window_width / 2
        cy = self.window_height / 2
        lines = message.count('\n') + 1
        box_h = max(40, 20 + lines * 18)
        box_w = min(self.window_width - 20, max(200, len(message) * 6))
        box = self.canvas.create_rectangle(
            cx - box_w // 2, cy - box_h, cx + box_w // 2, cy + box_h,
            fill='#222222', outline='#555555', width=1
        )
        msg_id = self.canvas.create_text(
            cx, cy,
            text=message,
            fill='#FFFFFF',
            font=('맑은 고딕', 11, 'bold'),
            justify='center',
            width=box_w - 20,
        )

        def remove():
            try:
                self.canvas.delete(box)
                self.canvas.delete(msg_id)
            except Exception:
                pass

        if self.root:
            self.root.after(duration, remove)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def update(self):
        """윈도우 업데이트 (이벤트 처리)"""
        if self.root:
            try:
                self.root.update()
            except tk.TclError:
                self.root = None
                self.canvas = None

    def mainloop(self):
        if self.root:
            self.root.mainloop()

    def close(self):
        self._polling = False
        if self._control_panel:
            self._control_panel.destroy()
            self._control_panel = None
        if self.root:
            try:
                if self._after_id is not None:
                    self.root.after_cancel(self._after_id)
                    self._after_id = None
                self.root.quit()
                self.root.destroy()
            except tk.TclError:
                pass
            finally:
                self.root = None
                self.canvas = None
                logger.info("오버레이 윈도우 종료")

    def reposition(self, x: int, y: int, w: int, h: int):
        """오버레이 위치/크기 재설정"""
        if self.root:
            try:
                self.window_x = x
                self.window_y = y
                self.window_width = w
                self.window_height = h
                self.root.geometry(f"{w}x{h}+{x}+{y}")
                if self.canvas:
                    self.canvas.config(width=w, height=h)
            except tk.TclError:
                pass

    def is_open(self) -> bool:
        if self.root is None:
            return False
        try:
            self.root.winfo_exists()
            return True
        except tk.TclError:
            self.root = None
            self.canvas = None
            return False


# 테스트 코드
if __name__ == "__main__":
    import time

    logging.basicConfig(level=logging.INFO)

    overlay = OverlayWindow(
        window_x=100, window_y=100,
        window_width=540, window_height=900
    )
    overlay.create_window()

    # 여러 가이드 표시
    guides = [
        GuideInfo(x=270, score=850, confidence=1.0, rank=0),
        GuideInfo(x=180, score=720, confidence=0.78, rank=1),
        GuideInfo(x=370, score=650, confidence=0.55, rank=2),
    ]
    overlay.update_guides(guides)
    overlay.update_status("MCTS depth=12 | 340 iters | score=1450")
    overlay.update_progress(0.85)

    def change():
        overlay.update_guides([
            GuideInfo(x=320, score=920, confidence=1.0, rank=0),
            GuideInfo(x=200, score=780, confidence=0.82, rank=1),
        ])
        overlay.update_status("MCTS depth=15 | 420 iters | score=1680")
        overlay.update_progress(1.0)

    overlay.root.after(4000, change)
    overlay.root.after(10000, overlay.close)

    overlay.mainloop()
