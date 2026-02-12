"""
로그 창 모듈 — LLM 분석 로그 + 디버깅 로그

별도 Toplevel 창으로 분석 결과와 디버그 로그를 실시간 표시합니다.
"""

import tkinter as tk
from tkinter import scrolledtext
import logging
from collections import deque

logger = logging.getLogger(__name__)


class AnalysisLogWindow:
    """LLM 분석 결과 + 양방향 대화 창"""

    def __init__(self, master: tk.Tk):
        self._win = tk.Toplevel(master)
        self._win.title("📊 LLM 분석 & 대화")
        self._win.attributes('-topmost', True)
        self._win.geometry("460x600")
        self._win.configure(bg='#1e1e2e')
        self._win.protocol("WM_DELETE_WINDOW", self.hide)

        self._chat_callback = None  # callable(msg) → str (비동기)

        # 헤더
        header = tk.Frame(self._win, bg='#1e1e2e')
        header.pack(fill='x', padx=8, pady=(8, 4))
        tk.Label(header, text="📊 LLM 분석 & 대화",
                 bg='#1e1e2e', fg='#89b4fa',
                 font=('맑은 고딕', 11, 'bold')).pack(side='left')
        tk.Button(header, text="🗑 지우기", bg='#313244', fg='#cdd6f4',
                  font=('맑은 고딕', 8), relief='flat', cursor='hand2',
                  command=self.clear).pack(side='right')

        # 스크롤 텍스트 (대화 + 분석 결과 표시)
        self._text = scrolledtext.ScrolledText(
            self._win, wrap='word',
            bg='#181825', fg='#cdd6f4',
            font=('맑은 고딕', 9),
            insertbackground='#cdd6f4',
            selectbackground='#45475a',
            relief='flat', borderwidth=0,
            state='disabled'
        )
        self._text.pack(fill='both', expand=True, padx=8, pady=(0, 4))

        # 태그 색상 정의
        self._text.tag_configure('header', foreground='#89b4fa', font=('맑은 고딕', 10, 'bold'))
        self._text.tag_configure('good', foreground='#a6e3a1')
        self._text.tag_configure('warn', foreground='#f9e2af')
        self._text.tag_configure('danger', foreground='#f38ba8')
        self._text.tag_configure('info', foreground='#94e2d5')
        self._text.tag_configure('dim', foreground='#6c7086')
        self._text.tag_configure('separator', foreground='#45475a')
        self._text.tag_configure('user_msg', foreground='#f9e2af', font=('맑은 고딕', 9, 'bold'))
        self._text.tag_configure('ai_msg', foreground='#a6e3a1')
        self._text.tag_configure('system', foreground='#6c7086', font=('맑은 고딕', 8))

        # 채팅 입력 영역
        chat_frame = tk.Frame(self._win, bg='#1e1e2e')
        chat_frame.pack(fill='x', padx=8, pady=(0, 8))

        self._chat_entry = tk.Entry(
            chat_frame,
            bg='#313244', fg='#cdd6f4',
            font=('맑은 고딕', 10),
            insertbackground='#cdd6f4',
            relief='flat', borderwidth=0
        )
        self._chat_entry.pack(side='left', fill='x', expand=True, ipady=6, padx=(0, 4))
        self._chat_entry.bind('<Return>', lambda e: self._on_send())

        self._btn_send = tk.Button(
            chat_frame, text="전송", bg='#89b4fa', fg='#1e1e2e',
            font=('맑은 고딕', 9, 'bold'), relief='flat', cursor='hand2',
            width=6, command=self._on_send
        )
        self._btn_send.pack(side='right', ipady=3)

        self._analysis_count = 0
        self._chat_busy = False

    def set_chat_callback(self, callback):
        """채팅 콜백 설정: callback(user_msg) → 백그라운드에서 호출됨"""
        self._chat_callback = callback

    def _on_send(self):
        """전송 버튼 / Enter 키"""
        msg = self._chat_entry.get().strip()
        if not msg or self._chat_busy:
            return
        self._chat_entry.delete(0, 'end')
        self._append_user_message(msg)

        if not self._chat_callback:
            self._append_ai_message("(채팅 기능이 연결되지 않았습니다)")
            return

        self._chat_busy = True
        self._btn_send.config(text="...", state='disabled')

        import threading
        def _do_chat():
            try:
                response = self._chat_callback(msg)
                self._win.after(0, lambda: self._append_ai_message(response))
            except Exception as e:
                self._win.after(0, lambda: self._append_ai_message(f"오류: {e}"))
            finally:
                self._win.after(0, self._chat_done)

        threading.Thread(target=_do_chat, daemon=True).start()

    def _chat_done(self):
        self._chat_busy = False
        self._btn_send.config(text="전송", state='normal')

    def _append_user_message(self, msg: str):
        """사용자 메시지 표시"""
        self._text.configure(state='normal')
        self._text.insert('end', f'\n🧑 나: ', 'user_msg')
        self._text.insert('end', f'{msg}\n', 'user_msg')
        self._text.configure(state='disabled')
        self._text.see('end')

    def _append_ai_message(self, msg: str):
        """AI 응답 표시"""
        self._text.configure(state='normal')
        self._text.insert('end', f'🤖 AI: ', 'ai_msg')
        self._text.insert('end', f'{msg}\n\n', 'ai_msg')
        self._text.configure(state='disabled')
        self._text.see('end')

    def add_analysis(self, result: dict, elapsed: float = 0.0):
        """분석 결과를 로그에 추가"""
        self._analysis_count += 1
        self._text.configure(state='normal')

        # 구분선
        if self._analysis_count > 1:
            self._text.insert('end', '─' * 50 + '\n', 'separator')

        # 헤더
        self._text.insert('end', f'#{self._analysis_count} 분석 결과', 'header')
        if elapsed > 0:
            self._text.insert('end', f'  ({elapsed:.1f}초)\n', 'dim')
        else:
            self._text.insert('end', '\n')

        # 드롭 위치
        drop_x = result.get('drop_x')
        confidence = result.get('confidence', 0.5)
        pct = int(confidence * 100)

        if pct >= 90:
            conf_desc = "매우 높음"
            conf_tag = 'good'
        elif pct >= 75:
            conf_desc = "높음"
            conf_tag = 'good'
        elif pct >= 60:
            conf_desc = "보통"
            conf_tag = 'warn'
        elif pct >= 40:
            conf_desc = "낮음"
            conf_tag = 'warn'
        else:
            conf_desc = "매우 낮음"
            conf_tag = 'danger'

        self._text.insert('end', f'  드롭: x={drop_x:.0f}\n' if drop_x else '  드롭: 없음\n')
        self._text.insert('end', f'  신뢰도: {pct}% ({conf_desc})\n', conf_tag)

        # 전략
        strategy = result.get('strategy', '')
        if strategy:
            self._text.insert('end', f'  전략: {strategy}\n', 'info')

        # 경로 검증
        path_check = result.get('path_check', '')
        if path_check:
            self._text.insert('end', f'  경로 검증: {path_check}\n', 'info')

        # 이유
        reason = result.get('reason', '')
        if reason:
            self._text.insert('end', f'  이유: {reason}\n')

        # 위험도
        risk = result.get('risk_level', 'safe')
        risk_map = {
            'safe': ('✅ 안전', 'good'),
            'warning': ('⚠️ 주의', 'warn'),
            'danger': ('🚨 위험', 'danger'),
        }
        risk_text, risk_tag = risk_map.get(risk, ('❓ 불명', 'dim'))
        self._text.insert('end', f'  위험도: {risk_text}\n', risk_tag)

        # 대안
        alt_x = result.get('alternative_x')
        alt_reason = result.get('alternative_reason', '')
        if alt_x is not None and alt_x != drop_x:
            self._text.insert('end', f'  대안: x={alt_x:.0f}', 'dim')
            if alt_reason:
                self._text.insert('end', f' — {alt_reason}', 'dim')
            self._text.insert('end', '\n')

        # 점수 + 동전
        score = result.get('game_score')
        if score is not None:
            self._text.insert('end', f'  현재 점수: {score}\n', 'info')
        current_coin = result.get('current_coin', '')
        if current_coin:
            self._text.insert('end', f'  현재 동전: {current_coin}\n', 'info')
        coins = result.get('coins', [])
        if coins:
            self._text.insert('end', f'  감지된 동전: {len(coins)}개\n', 'dim')

        self._text.insert('end', '\n')
        self._text.configure(state='disabled')
        self._text.see('end')

    def clear(self):
        self._text.configure(state='normal')
        self._text.delete('1.0', 'end')
        self._text.configure(state='disabled')
        self._analysis_count = 0

    def show(self):
        try:
            self._win.deiconify()
            self._win.lift()
        except tk.TclError:
            pass

    def hide(self):
        try:
            self._win.withdraw()
        except tk.TclError:
            pass

    def destroy(self):
        try:
            self._win.destroy()
        except Exception:
            pass


class DebugLogWindow:
    """Python logging 출력을 실시간으로 보여주는 디버그 창"""

    def __init__(self, master: tk.Tk):
        self._win = tk.Toplevel(master)
        self._win.title("🔧 디버그 로그")
        self._win.attributes('-topmost', True)
        self._win.geometry("550x400")
        self._win.configure(bg='#1e1e2e')
        self._win.protocol("WM_DELETE_WINDOW", self.hide)

        # 헤더
        header = tk.Frame(self._win, bg='#1e1e2e')
        header.pack(fill='x', padx=8, pady=(8, 4))
        tk.Label(header, text="🔧 디버그 로그",
                 bg='#1e1e2e', fg='#f9e2af',
                 font=('맑은 고딕', 11, 'bold')).pack(side='left')
        tk.Button(header, text="🗑 지우기", bg='#313244', fg='#cdd6f4',
                  font=('맑은 고딕', 8), relief='flat', cursor='hand2',
                  command=self.clear).pack(side='right')

        # 스크롤 텍스트
        self._text = scrolledtext.ScrolledText(
            self._win, wrap='word',
            bg='#11111b', fg='#a6adc8',
            font=('Consolas', 8),
            insertbackground='#cdd6f4',
            selectbackground='#45475a',
            relief='flat', borderwidth=0,
            state='disabled'
        )
        self._text.pack(fill='both', expand=True, padx=8, pady=(0, 8))

        # 태그
        self._text.tag_configure('DEBUG', foreground='#6c7086')
        self._text.tag_configure('INFO', foreground='#94e2d5')
        self._text.tag_configure('WARNING', foreground='#f9e2af')
        self._text.tag_configure('ERROR', foreground='#f38ba8')
        self._text.tag_configure('CRITICAL', foreground='#f38ba8', font=('Consolas', 8, 'bold'))

        # logging 핸들러 등록
        self._handler = _TkTextHandler(self)
        self._handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%H:%M:%S'
        ))
        logging.getLogger().addHandler(self._handler)

        # 버퍼 (창이 닫혀있을 때도 최근 로그 유지)
        self._buffer = deque(maxlen=500)

    def append_log(self, msg: str, level: str = 'INFO'):
        """로그 메시지 추가"""
        self._buffer.append((msg, level))
        try:
            self._text.configure(state='normal')
            self._text.insert('end', msg + '\n', level)
            # 최대 줄 수 제한
            line_count = int(self._text.index('end-1c').split('.')[0])
            if line_count > 500:
                self._text.delete('1.0', f'{line_count - 400}.0')
            self._text.configure(state='disabled')
            self._text.see('end')
        except (tk.TclError, Exception):
            pass

    def clear(self):
        try:
            self._text.configure(state='normal')
            self._text.delete('1.0', 'end')
            self._text.configure(state='disabled')
            self._buffer.clear()
        except Exception:
            pass

    def show(self):
        try:
            self._win.deiconify()
            self._win.lift()
        except tk.TclError:
            pass

    def hide(self):
        try:
            self._win.withdraw()
        except tk.TclError:
            pass

    def destroy(self):
        try:
            logging.getLogger().removeHandler(self._handler)
        except Exception:
            pass
        try:
            self._win.destroy()
        except Exception:
            pass


class _TkTextHandler(logging.Handler):
    """logging.Handler → DebugLogWindow 연결"""

    def __init__(self, debug_win: DebugLogWindow):
        super().__init__()
        self._win = debug_win

    def emit(self, record):
        try:
            msg = self.format(record)
            level = record.levelname
            self._win.append_log(msg, level)
        except Exception:
            pass
