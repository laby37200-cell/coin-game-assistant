"""
오버레이 UI

tkinter를 사용하여 게임 화면 위에 투명한 가이드 라인을 표시합니다.
"""

import tkinter as tk
import logging
from typing import Optional, Tuple


logger = logging.getLogger(__name__)


class OverlayWindow:
    """투명 오버레이 윈도우 클래스"""
    
    def __init__(
        self,
        window_x: int,
        window_y: int,
        window_width: int,
        window_height: int,
        opacity: float = 0.7,
        line_color: str = "#00FF00",
        line_width: int = 3,
        text_color: str = "#FFFFFF",
        text_size: int = 16
    ):
        """
        Args:
            window_x, window_y: 오버레이 윈도우 위치
            window_width, window_height: 오버레이 윈도우 크기
            opacity: 투명도 (0~1)
            line_color: 가이드 라인 색상
            line_width: 가이드 라인 두께
            text_color: 텍스트 색상
            text_size: 텍스트 크기
        """
        self.window_x = window_x
        self.window_y = window_y
        self.window_width = window_width
        self.window_height = window_height
        self.opacity = opacity
        self.line_color = line_color
        self.line_width = line_width
        self.text_color = text_color
        self.text_size = text_size
        
        # tkinter 윈도우
        self.root = None
        self.canvas = None
        
        # 현재 표시 중인 가이드 정보
        self.current_guide_x = None
        self.current_score = None
        
        # 그래픽 객체 ID
        self.line_id = None
        self.text_id = None
        
        logger.info(f"OverlayWindow 초기화: {window_width}x{window_height} at ({window_x}, {window_y})")
    
    def create_window(self):
        """오버레이 윈도우 생성"""
        # tkinter 루트 윈도우
        self.root = tk.Tk()
        
        # 윈도우 설정
        self.root.title("Coin Game Assistant")
        self.root.geometry(f"{self.window_width}x{self.window_height}+{self.window_x}+{self.window_y}")
        
        # 항상 최상위
        self.root.attributes('-topmost', True)
        
        # 투명 배경 (Windows)
        try:
            self.root.attributes('-transparentcolor', 'black')
            self.root.attributes('-alpha', self.opacity)
        except:
            logger.warning("투명 배경 설정 실패 (일부 플랫폼에서는 지원되지 않음)")
        
        # 캔버스 생성
        self.canvas = tk.Canvas(
            self.root,
            width=self.window_width,
            height=self.window_height,
            bg='black',
            highlightthickness=0
        )
        self.canvas.pack()
        
        # 종료 이벤트
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        
        logger.info("오버레이 윈도우 생성 완료")
    
    def update_guide(self, guide_x: float, score: float):
        """
        가이드 라인 업데이트
        
        Args:
            guide_x: 가이드 x 좌표 (게임 영역 기준)
            score: 예상 점수
        """
        if not self.canvas:
            logger.warning("캔버스가 생성되지 않았습니다.")
            return
        
        # 이전 가이드 제거
        self.clear_guide()
        
        # 수직선 그리기
        self.line_id = self.canvas.create_line(
            guide_x, 0,
            guide_x, self.window_height,
            fill=self.line_color,
            width=self.line_width
        )
        
        # 텍스트 그리기
        text = f"여기에 떨어트리세요!\n예상 점수: {score:.0f}"
        
        # 텍스트 위치 (라인 위, 중앙)
        text_x = guide_x
        text_y = 50
        
        # 텍스트가 화면 밖으로 나가지 않도록 조정
        if text_x < 100:
            text_x = 100
        elif text_x > self.window_width - 100:
            text_x = self.window_width - 100
        
        # 그림자 효과 (검은색 배경)
        self.canvas.create_text(
            text_x + 2, text_y + 2,
            text=text,
            fill='black',
            font=('Arial', self.text_size, 'bold'),
            justify='center'
        )
        
        # 메인 텍스트
        self.text_id = self.canvas.create_text(
            text_x, text_y,
            text=text,
            fill=self.text_color,
            font=('Arial', self.text_size, 'bold'),
            justify='center'
        )
        
        # 현재 가이드 정보 저장
        self.current_guide_x = guide_x
        self.current_score = score
        
        logger.debug(f"가이드 업데이트: x={guide_x:.1f}, score={score:.1f}")
    
    def clear_guide(self):
        """가이드 라인 제거"""
        if self.canvas:
            self.canvas.delete('all')
            self.line_id = None
            self.text_id = None
            self.current_guide_x = None
            self.current_score = None
    
    def show_message(self, message: str, duration: int = 3000):
        """
        일시적인 메시지 표시
        
        Args:
            message: 표시할 메시지
            duration: 표시 시간 (밀리초)
        """
        if not self.canvas:
            return
        
        # 메시지 텍스트
        msg_id = self.canvas.create_text(
            self.window_width / 2,
            self.window_height / 2,
            text=message,
            fill=self.text_color,
            font=('Arial', 20, 'bold'),
            justify='center'
        )
        
        # 일정 시간 후 제거
        self.root.after(duration, lambda: self.canvas.delete(msg_id))
    
    def update(self):
        """윈도우 업데이트 (이벤트 처리)"""
        if self.root:
            try:
                self.root.update()
            except tk.TclError:
                self.root = None
                self.canvas = None
    
    def mainloop(self):
        """메인 루프 시작 (블로킹)"""
        if self.root:
            self.root.mainloop()
    
    def close(self):
        """윈도우 종료"""
        if self.root:
            try:
                self.root.quit()
                self.root.destroy()
            except tk.TclError:
                pass
            finally:
                self.root = None
                self.canvas = None
                logger.info("오버레이 윈도우 종료")
    
    def is_open(self) -> bool:
        """윈도우 열림 상태 확인"""
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
    
    print("=== 오버레이 UI 테스트 ===\n")
    print("3초 후 오버레이 윈도우가 표시됩니다...")
    time.sleep(3)
    
    # OverlayWindow 생성
    overlay = OverlayWindow(
        window_x=100,
        window_y=100,
        window_width=600,
        window_height=800
    )
    
    # 윈도우 생성
    overlay.create_window()
    print("✅ 오버레이 윈도우 생성 완료")
    
    # 가이드 라인 표시
    overlay.update_guide(guide_x=300, score=150.5)
    print("✅ 가이드 라인 표시")
    
    # 메시지 표시
    overlay.show_message("테스트 메시지", duration=2000)
    
    # 5초 후 가이드 변경
    def change_guide():
        overlay.update_guide(guide_x=450, score=200.0)
        print("✅ 가이드 라인 변경")
    
    overlay.root.after(5000, change_guide)
    
    # 10초 후 종료
    def close_window():
        print("✅ 오버레이 윈도우 종료")
        overlay.close()
    
    overlay.root.after(10000, close_window)
    
    # 메인 루프
    print("\n오버레이 윈도우가 10초 동안 표시됩니다...")
    overlay.mainloop()
    
    print("\n✅ 테스트 완료")
