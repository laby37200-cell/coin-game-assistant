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
    __slots__ = ('x', 'score', 'confidence', 'rank')

    def __init__(self, x: float, score: float, confidence: float, rank: int):
        self.x = x
        self.score = score
        self.confidence = confidence  # 0.0 ~ 1.0
        self.rank = rank              # 0 = best


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

        # 스레드 안전 큐 — 백그라운드 스레드에서 UI 업데이트 요청
        self._queue = queue.Queue()

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

        # 화살표 키 조정
        self._adjust_target = 0  # 0=wall_left, 1=wall_right, 2=ceiling, 3=floor
        self._adjust_names = ['Wall-L', 'Wall-R', 'Ceiling', 'Floor']
        self._adjust_step = 3  # pixels per keypress
        self._bounds_callback = None  # callable(wall_l, wall_r, ceil, floor)
        self._analyze_callback = None   # callable() — 수동 분석 트리거
        self._auto_mode = False          # 기본: 수동 분석
        self._bounds_editing = False     # 경계 편집 모드 (Ctrl+Enter로 토글)

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
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
            style = ctypes.windll.user32.GetWindowLongW(hwnd, -20)  # GWL_EXSTYLE
            # WS_EX_LAYERED 만 설정 (WS_EX_TRANSPARENT 는 렌더링 문제 유발 가능)
            style |= 0x80000  # WS_EX_LAYERED
            ctypes.windll.user32.SetWindowLongW(hwnd, -20, style)
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

        # 화살표 키로 벽/천장 조정 (편집모드일 때만 동작)
        self.root.bind('<Left>', lambda e: self._adjust_horizontal(-self._adjust_step))
        self.root.bind('<Right>', lambda e: self._adjust_horizontal(self._adjust_step))
        self.root.bind('<Up>', lambda e: self._adjust_vertical(-self._adjust_step))
        self.root.bind('<Down>', lambda e: self._adjust_vertical(self._adjust_step))
        # Tab = 수동 분석 (break로 포커스 이동 방지)
        self.root.bind('<Tab>', self._on_tab)
        # Ctrl+Tab = 자동/수동 모드 토글
        self.root.bind('<Control-Tab>', lambda e: self._toggle_auto_mode())
        # Enter = 편집모드에서 다음 선으로 이동
        self.root.bind('<Return>', lambda e: self._next_bound())
        # Ctrl+Enter = 경계 편집 모드 토글 (저장+잠금)
        self.root.bind('<Control-Return>', lambda e: self._toggle_editing())
        self.root.bind('b', lambda e: self._toggle_bounds())
        # Shift+Arrow = 큰 조정 (10px)
        self.root.bind('<Shift-Left>', lambda e: self._adjust_horizontal(-10))
        self.root.bind('<Shift-Right>', lambda e: self._adjust_horizontal(10))
        self.root.bind('<Shift-Up>', lambda e: self._adjust_vertical(-10))
        self.root.bind('<Shift-Down>', lambda e: self._adjust_vertical(10))

        # 포커스 강제 설정 (overrideredirect 윈도우는 포커스를 잃기 쉬움)
        self.root.focus_force()
        self.canvas.focus_set()

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
        except Exception:
            pass
        if self.root and self._polling:
            try:
                # 포커스 유지 (overrideredirect 윈도우는 포커스를 잃기 쉬움)
                if self.root.focus_get() is None:
                    self.root.focus_force()
                self._after_id = self.root.after(50, self._poll_queue)
            except tk.TclError:
                pass

    # ------------------------------------------------------------------ #
    # Public API (thread-safe — can be called from any thread)
    # ------------------------------------------------------------------ #
    def update_guide(self, guide_x: float, score: float):
        """단일 가이드 업데이트 (하위 호환)"""
        self.update_guides([GuideInfo(x=guide_x, score=score, confidence=1.0, rank=0)])

    def update_guides(self, guides: List[GuideInfo]):
        """여러 가이드 라인 업데이트 (스레드 안전)"""
        self._queue.put((self._MSG_GUIDES, guides))

    def update_status(self, text: str):
        """상태 텍스트 업데이트 (스레드 안전)"""
        self._queue.put((self._MSG_STATUS, text))

    def update_progress(self, progress: float):
        """진행도 업데이트 0.0~1.0 (스레드 안전)"""
        self._queue.put((self._MSG_PROGRESS, max(0.0, min(1.0, progress))))

    def show_message(self, message: str, duration: int = 3000):
        """일시적 메시지 표시 (스레드 안전)"""
        self._queue.put((self._MSG_MESSAGE, (message, duration)))

    def clear_guide(self):
        """가이드 제거"""
        self._guides = []
        self._queue.put((self._MSG_GUIDES, []))

    def update_bounds(self, wall_left: float, wall_right: float,
                      ceiling_y: float, floor_y: float):
        """벽/천장/바닥 경계 업데이트 (스레드 안전)"""
        self._queue.put((self._MSG_BOUNDS, (wall_left, wall_right, ceiling_y, floor_y)))

    def set_bounds_callback(self, callback):
        """경계 조정 시 호출될 콜백 설정: callback(wall_l, wall_r, ceil, floor)"""
        self._bounds_callback = callback

    def set_analyze_callback(self, callback):
        """수동 분석 트리거 콜백 설정: callback()"""
        self._analyze_callback = callback

    @property
    def auto_mode(self) -> bool:
        return self._auto_mode

    # ------------------------------------------------------------------ #
    # Keyboard boundary adjustment (메인 스레드)
    # ------------------------------------------------------------------ #
    def _adjust_horizontal(self, delta: int):
        """Left/Right 키 — 편집모드 + 벽 선택 시에만 이동"""
        if not self._bounds_editing:
            return
        t = self._adjust_target
        if t == 0:
            self._wall_left = max(0, self._wall_left + delta)
        elif t == 1:
            self._wall_right = min(self.window_width, self._wall_right + delta)
        else:
            return
        self._redraw()

    def _adjust_vertical(self, delta: int):
        """Up/Down 키 — 편집모드 + 천장/바닥 선택 시에만 이동"""
        if not self._bounds_editing:
            return
        t = self._adjust_target
        if t == 2:
            self._ceiling_y = max(0, self._ceiling_y + delta)
        elif t == 3:
            self._floor_y = min(self.window_height, self._floor_y + delta)
        else:
            return
        self._redraw()

    def _notify_bounds(self):
        """콜백으로 main에 경계값 변경 알림 (저장 포함)"""
        if self._bounds_callback:
            self._bounds_callback(
                self._wall_left, self._wall_right,
                self._ceiling_y, self._floor_y)

    def _next_bound(self):
        """Enter 키 — 편집모드에서 다음 선으로 이동 (값은 그대로 유지)"""
        if not self._bounds_editing:
            return
        self._adjust_target = (self._adjust_target + 1) % 4
        name = self._adjust_names[self._adjust_target]
        logger.info(f"Next bound: {name}")
        self._redraw()

    def _toggle_editing(self):
        """Ctrl+Enter — 경계 편집 모드 토글. 나갈 때 저장."""
        self._bounds_editing = not self._bounds_editing
        if self._bounds_editing:
            self._adjust_target = 0
            logger.info("경계 편집 모드 진입")
        else:
            # 편집 종료 → 저장
            self._notify_bounds()
            logger.info("경계 편집 모드 종료 — 저장 완료")
        self._redraw()

    def _on_tab(self, event):
        """Tab 키 — 수동 분석 트리거 (break로 포커스 이동 방지)"""
        if self._analyze_callback:
            self._analyze_callback()
            logger.info("수동 분석 트리거")
        return 'break'

    def _toggle_auto_mode(self):
        """Ctrl+Tab — 자동/수동 분석 모드 토글"""
        self._auto_mode = not self._auto_mode
        mode = "자동" if self._auto_mode else "수동"
        logger.info(f"분석 모드: {mode}")
        self._redraw()

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

        editing = self._bounds_editing
        t = self._adjust_target

        # 색상: 편집모드 + 선택된 것만 밝게, 나머지는 어둡게
        def col(idx, base):
            if not editing:
                return '#444444'  # 잠금 상태 — 모두 어둡게
            return base if idx == t else '#555555'

        # 편집모드일 때 선택된 선은 굵게
        def lw(idx):
            return 3 if (editing and idx == t) else 2

        wl = self._wall_left
        wr = self._wall_right
        cy = self._ceiling_y
        fy = self._floor_y

        # 왼쪽 벽
        self.canvas.create_line(
            wl, 0, wl, self.window_height,
            fill=col(0, '#00CCFF'), width=lw(0), dash=(4, 4))
        self.canvas.create_text(
            wl + 3, 15, text=f'L:{wl:.0f}', fill=col(0, '#00CCFF'),
            font=('Consolas', 8), anchor='w')

        # 오른쪽 벽
        self.canvas.create_line(
            wr, 0, wr, self.window_height,
            fill=col(1, '#00CCFF'), width=lw(1), dash=(4, 4))
        self.canvas.create_text(
            wr - 3, 15, text=f'R:{wr:.0f}', fill=col(1, '#00CCFF'),
            font=('Consolas', 8), anchor='e')

        # 천장 (게임오버 라인)
        self.canvas.create_line(
            0, cy, self.window_width, cy,
            fill=col(2, '#FF4444'), width=lw(2), dash=(6, 3))
        self.canvas.create_text(
            self.window_width - 5, cy + 10, text=f'Ceil:{cy:.0f}',
            fill=col(2, '#FF4444'), font=('Consolas', 8), anchor='e')

        # 바닥
        self.canvas.create_line(
            0, fy, self.window_width, fy,
            fill=col(3, '#44FF44'), width=lw(3), dash=(6, 3))
        self.canvas.create_text(
            self.window_width - 5, fy - 10, text=f'Floor:{fy:.0f}',
            fill=col(3, '#44FF44'), font=('Consolas', 8), anchor='e')

        # 상단 안내 텍스트
        mode_str = '자동' if self._auto_mode else '수동'
        mode_col = '#00FF00' if self._auto_mode else '#FFAA00'

        if editing:
            name = self._adjust_names[t]
            if t in (0, 1):
                hint = f'[←→] ±{self._adjust_step}px  |  [Shift] ±10px'
            else:
                hint = f'[↑↓] ±{self._adjust_step}px  |  [Shift] ±10px'
            self.canvas.create_text(
                self.window_width / 2, 5,
                text=f'✏ 편집: {name}  |  {hint}  |  [Enter] 다음선  |  [Ctrl+Enter] 저장+잠금',
                fill='#FF8800', font=('Consolas', 9, 'bold'), anchor='n')
        else:
            self.canvas.create_text(
                self.window_width / 2, 5,
                text=f'🔒 경계 잠금  |  [Ctrl+Enter] 편집  |  [b] 숨기기',
                fill='#888888', font=('Consolas', 9, 'bold'), anchor='n')
        self.canvas.create_text(
            self.window_width / 2, 20,
            text=f'[Tab] 분석  |  [Ctrl+Tab] 모드: {mode_str}',
            fill=mode_col, font=('Consolas', 9, 'bold'), anchor='n')

    def _draw_guide_lines(self):
        """현재 저장된 가이드 라인들을 캔버스에 그리기"""
        if not self.canvas or not self._guides:
            return

        for g in reversed(self._guides):  # 낮은 순위부터 그려서 최고가 위에
            color = confidence_color(g.confidence)
            width = self.line_width + (2 if g.rank == 0 else 0)

            # 캔버스 범위 내로 클램핑
            gx = max(5, min(g.x, self.window_width - 5))

            # 수직 가이드 라인
            self.canvas.create_line(
                gx, 80, gx, self.window_height - 20,
                fill=color, width=width, dash=(6, 3) if g.rank > 0 else ()
            )

            # 라인 상단에 삼각형 마커
            self.canvas.create_polygon(
                gx - 8, 80, gx + 8, 80, gx, 95,
                fill=color, outline=color
            )

            # 확률 텍스트
            pct = int(g.confidence * 100)
            label = f"{pct}%"
            tx = gx
            if tx < 40:
                tx = 40
            elif tx > self.window_width - 40:
                tx = self.window_width - 40

            self.canvas.create_text(
                tx, 70,
                text=label,
                fill=color,
                font=('Consolas', 11, 'bold'),
            )

        # 최고 가이드 정보 텍스트
        best = self._guides[0] if self._guides else None
        if best:
            info = f"DROP HERE  (score: {best.score:.0f})"
            tx = max(5, min(best.x, self.window_width - 5))
            if tx < 80:
                tx = 80
            elif tx > self.window_width - 80:
                tx = self.window_width - 80

            # 텍스트 배경 박스
            self.canvas.create_rectangle(
                tx - 75, 38, tx + 75, 60,
                fill='#222222', outline=confidence_color(best.confidence), width=1
            )
            self.canvas.create_text(
                tx, 49,
                text=info,
                fill='#FFFFFF',
                font=('Consolas', 10, 'bold'),
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

        # 반투명 배경 박스
        cx = self.window_width / 2
        cy = self.window_height / 2
        box = self.canvas.create_rectangle(
            cx - 140, cy - 30, cx + 140, cy + 30,
            fill='#222222', outline='#555555', width=1
        )
        msg_id = self.canvas.create_text(
            cx, cy,
            text=message,
            fill='#FFFFFF',
            font=('Consolas', 13, 'bold'),
            justify='center'
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
