"""
컨트롤 패널 — 별도 tkinter 창으로 버튼 UI 제공

exe 환경에서 키보드 훅이 안 먹히므로, 클릭 가능한 버튼으로 조작합니다.
오버레이 창 옆에 작은 패널이 열립니다.
"""

import tkinter as tk
import logging

logger = logging.getLogger(__name__)

# 지연 import 방지를 위해 모듈 레벨에서 None으로 초기화
_AnalysisLogWindow = None
_DebugLogWindow = None

def _ensure_log_imports():
    global _AnalysisLogWindow, _DebugLogWindow
    if _AnalysisLogWindow is None:
        from ui.log_windows import AnalysisLogWindow, DebugLogWindow
        _AnalysisLogWindow = AnalysisLogWindow
        _DebugLogWindow = DebugLogWindow


class ControlPanel:
    """별도 Toplevel 컨트롤 패널"""

    def __init__(self, master: tk.Tk, overlay_ref):
        """
        master: 오버레이의 root (Tk)
        overlay_ref: OverlayWindow 인스턴스
        """
        self._overlay = overlay_ref
        self._master = master
        self._win = tk.Toplevel(master)
        self._win.title("컨트롤")
        self._win.attributes('-topmost', True)
        self._win.resizable(False, False)
        self._win.protocol("WM_DELETE_WINDOW", self._on_close)

        # 오버레이 왼쪽에 배치
        px = max(0, overlay_ref.window_x - 160)
        py = overlay_ref.window_y
        self._win.geometry(f"155x560+{px}+{py}")

        # 로그 창 참조
        self._analysis_log = None
        self._debug_log = None

        self._build_ui()
        logger.info("컨트롤 패널 생성")

    def _build_ui(self):
        bg = '#1e1e2e'
        fg = '#cdd6f4'
        btn_bg = '#313244'
        btn_active = '#45475a'
        accent = '#89b4fa'
        green = '#a6e3a1'
        yellow = '#f9e2af'
        red = '#f38ba8'

        self._win.configure(bg=bg)

        # 타이틀
        tk.Label(
            self._win, text="🎮 동전게임\n   가이드",
            bg=bg, fg=accent,
            font=('맑은 고딕', 12, 'bold'),
            justify='center'
        ).pack(pady=(10, 5))

        # 구분선
        tk.Frame(self._win, bg='#585b70', height=1).pack(fill='x', padx=10, pady=5)

        # ── 분석 버튼 (가장 크게) ──
        self._btn_analyze = tk.Button(
            self._win, text="▶  분석 시작",
            bg=green, fg='#1e1e2e',
            activebackground='#94e2d5', activeforeground='#1e1e2e',
            font=('맑은 고딕', 13, 'bold'),
            relief='flat', cursor='hand2',
            height=2,
            command=self._on_analyze
        )
        self._btn_analyze.pack(fill='x', padx=10, pady=(5, 3))

        # ── 자동 터치 모드 토글 ──
        self._auto_touch_on = False
        self._btn_auto = tk.Button(
            self._win, text="🤖 자동 터치 OFF",
            bg=btn_bg, fg='#f38ba8',
            activebackground=btn_active, activeforeground='#f38ba8',
            font=('맑은 고딕', 11, 'bold'),
            relief='flat', cursor='hand2',
            command=self._on_toggle_auto_touch
        )
        self._btn_auto.pack(fill='x', padx=10, pady=(3, 3))

        # 구분선
        tk.Frame(self._win, bg='#585b70', height=1).pack(fill='x', padx=10, pady=5)

        # ── 경계값 입력 필드 (4개) ──
        bounds_lbl_w = 5
        entry_w = 5
        ov = self._overlay

        def _make_bound_row(parent, label_text, init_val):
            row = tk.Frame(parent, bg=bg)
            row.pack(fill='x', padx=10, pady=1)
            tk.Label(row, text=label_text, bg=bg, fg=yellow,
                     font=('Consolas', 8, 'bold'), width=bounds_lbl_w,
                     anchor='w').pack(side='left')
            var = tk.StringVar(value=str(int(init_val)))
            ent = tk.Entry(row, textvariable=var,
                          bg='#313244', fg='#cdd6f4', insertbackground='#cdd6f4',
                          font=('Consolas', 9), relief='flat', width=entry_w)
            ent.pack(side='left', padx=2, ipady=1)
            ent.bind('<Return>', lambda e: self._on_apply_bounds())
            return var

        tk.Label(self._win, text="─ 경계값 설정 ─", bg=bg, fg=yellow,
                 font=('맑은 고딕', 9, 'bold')).pack(pady=(0, 2))

        self._var_wl = _make_bound_row(self._win, "L 벽", ov._wall_left)
        self._var_wr = _make_bound_row(self._win, "R 벽", ov._wall_right)
        self._var_ceil = _make_bound_row(self._win, "천장", ov._ceiling_y)
        self._var_floor = _make_bound_row(self._win, "바닥", ov._floor_y)

        tk.Button(
            self._win, text="경계 적용", bg=accent, fg='#1e1e2e',
            font=('맑은 고딕', 9, 'bold'), relief='flat', cursor='hand2',
            command=self._on_apply_bounds
        ).pack(fill='x', padx=10, pady=(3, 2))

        # 구분선
        tk.Frame(self._win, bg='#585b70', height=1).pack(fill='x', padx=10, pady=5)

        # ── 경계선 표시/숨기기 ──
        tk.Button(
            self._win, text="👁 경계선 토글",
            bg=btn_bg, fg=fg,
            activebackground=btn_active, activeforeground=fg,
            font=('맑은 고딕', 9), relief='flat', cursor='hand2',
            command=self._on_toggle_bounds
        ).pack(fill='x', padx=10, pady=2)

        # ── 분석 로그 창 ──
        tk.Button(
            self._win, text="📊 분석 & 대화",
            bg=btn_bg, fg='#94e2d5',
            activebackground=btn_active, activeforeground='#94e2d5',
            font=('맑은 고딕', 9), relief='flat', cursor='hand2',
            command=self._on_show_analysis_log
        ).pack(fill='x', padx=10, pady=2)

        # ── 디버그 로그 창 ──
        tk.Button(
            self._win, text="🔧 디버그 로그",
            bg=btn_bg, fg='#6c7086',
            activebackground=btn_active, activeforeground='#a6adc8',
            font=('맑은 고딕', 9), relief='flat', cursor='hand2',
            command=self._on_show_debug_log
        ).pack(fill='x', padx=10, pady=2)

        # ── 종료 ──
        tk.Button(
            self._win, text="✕ 종료",
            bg=btn_bg, fg=red,
            activebackground='#45475a', activeforeground=red,
            font=('맑은 고딕', 9, 'bold'), relief='flat', cursor='hand2',
            command=self._on_close
        ).pack(fill='x', padx=10, pady=(8, 10))

        # 상태 라벨
        self._lbl_status = tk.Label(
            self._win, text="대기 중",
            bg=bg, fg='#6c7086',
            font=('맑은 고딕', 8)
        )
        self._lbl_status.pack(side='bottom', pady=(0, 5))

    # ── 콜백 ──
    def _on_analyze(self):
        if self._overlay and self._overlay._analyze_callback:
            self._overlay._analyze_callback()
            self._lbl_status.config(text="분석 요청됨...")
            logger.info("컨트롤 패널: 분석 트리거")

    def _on_apply_bounds(self):
        """경계값 4개 적용"""
        ov = self._overlay
        try:
            wl = float(self._var_wl.get().strip())
            wr = float(self._var_wr.get().strip())
            cy = float(self._var_ceil.get().strip())
            fy = float(self._var_floor.get().strip())
            # 범위 제한
            wl = max(0, min(ov.window_width, wl))
            wr = max(wl + 20, min(ov.window_width, wr))
            cy = max(0, min(ov.window_height, cy))
            fy = max(cy + 20, min(ov.window_height, fy))
            ov._wall_left = wl
            ov._wall_right = wr
            ov._ceiling_y = cy
            ov._floor_y = fy
            ov._bounds_visible = True
            ov._notify_bounds()
            ov._redraw()
            # 표시 값 동기화
            self._var_wl.set(str(int(wl)))
            self._var_wr.set(str(int(wr)))
            self._var_ceil.set(str(int(cy)))
            self._var_floor.set(str(int(fy)))
            self._lbl_status.config(text=f"L={int(wl)} R={int(wr)} C={int(cy)} F={int(fy)}")
            logger.info(f"경계값 적용: L={wl:.0f} R={wr:.0f} C={cy:.0f} F={fy:.0f}")
        except ValueError:
            self._lbl_status.config(text="숫자를 입력하세요")

    def _on_toggle_auto_touch(self):
        """자동 터치 모드 토글"""
        self._auto_touch_on = not self._auto_touch_on
        if self._auto_touch_on:
            self._btn_auto.config(text="🤖 자동 터치 ON", bg='#a6e3a1', fg='#1e1e2e',
                                  activebackground='#94e2d5', activeforeground='#1e1e2e')
            self._lbl_status.config(text="자동 터치 모드 ON")
        else:
            self._btn_auto.config(text="🤖 자동 터치 OFF", bg='#313244', fg='#f38ba8',
                                  activebackground='#45475a', activeforeground='#f38ba8')
            self._lbl_status.config(text="자동 터치 모드 OFF")
        # main_llm 콜백 호출
        if self._overlay and self._overlay._auto_touch_callback:
            self._overlay._auto_touch_callback()

    def update_bounds_display(self, wl, wr, cy, fy):
        """외부에서 경계값 표시 업데이트"""
        try:
            self._var_wl.set(str(int(wl)))
            self._var_wr.set(str(int(wr)))
            self._var_ceil.set(str(int(cy)))
            self._var_floor.set(str(int(fy)))
        except Exception:
            pass

    def _on_toggle_bounds(self):
        self._overlay._toggle_bounds()

    def _on_show_analysis_log(self):
        _ensure_log_imports()
        if self._analysis_log is None:
            self._analysis_log = _AnalysisLogWindow(self._master)
            # 채팅 콜백 연결: overlay → main_llm → llm_advisor.chat()
            if self._overlay and hasattr(self._overlay, '_chat_callback') and self._overlay._chat_callback:
                self._analysis_log.set_chat_callback(self._overlay._chat_callback)
        self._analysis_log.show()

    def _on_show_debug_log(self):
        _ensure_log_imports()
        if self._debug_log is None:
            self._debug_log = _DebugLogWindow(self._master)
        self._debug_log.show()

    def add_analysis_result(self, result: dict, elapsed: float = 0.0):
        """분석 결과를 분석 로그 창에 추가"""
        if self._analysis_log is not None:
            self._analysis_log.add_analysis(result, elapsed)

    def _on_close(self):
        self._overlay.close()

    def update_status(self, text: str):
        """외부에서 상태 텍스트 업데이트"""
        try:
            self._lbl_status.config(text=text)
        except Exception:
            pass

    def destroy(self):
        if self._analysis_log:
            self._analysis_log.destroy()
            self._analysis_log = None
        if self._debug_log:
            self._debug_log.destroy()
            self._debug_log = None
        try:
            self._win.destroy()
        except Exception:
            pass
