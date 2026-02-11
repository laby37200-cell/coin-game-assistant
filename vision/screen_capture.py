"""
화면 캡처 모듈

MuMu Player 창을 찾아서 게임 화면을 실시간으로 캡처합니다.
"""

import time
import numpy as np
from PIL import Image
from typing import Optional, Tuple
import logging

try:
    import mss
    import pygetwindow as gw
except ImportError as e:
    logging.warning(f"Import warning: {e}. Some features may not work on headless systems.")


logger = logging.getLogger(__name__)


class ScreenCapture:
    """MuMu Player 화면 캡처 클래스"""
    
    def __init__(self, window_title_pattern: str = "MuMu"):
        """
        Args:
            window_title_pattern: MuMu Player 창 제목에 포함될 패턴
        """
        self.window_title_pattern = window_title_pattern
        self.window = None
        self.game_area = None  # (x, y, width, height)
        self.last_screenshot = None
        self.last_capture_time = 0
        
        logger.info(f"ScreenCapture 초기화: 창 패턴 = '{window_title_pattern}'")
    
    def find_window(self) -> bool:
        """
        MuMu Player 창 찾기
        
        Returns:
            성공 여부
        """
        try:
            windows = gw.getWindowsWithTitle(self.window_title_pattern)
            
            if not windows:
                logger.warning(f"'{self.window_title_pattern}' 패턴을 가진 창을 찾을 수 없습니다.")
                return False
            
            # 첫 번째 매칭 창 선택
            self.window = windows[0]
            logger.info(f"창 발견: {self.window.title} at ({self.window.left}, {self.window.top})")
            
            # 게임 영역 설정 (전체 창)
            self.game_area = {
                'left': self.window.left,
                'top': self.window.top,
                'width': self.window.width,
                'height': self.window.height
            }
            
            return True
            
        except Exception as e:
            logger.error(f"창 찾기 실패: {e}")
            return False
    
    def set_game_area(self, x: int, y: int, width: int, height: int):
        """
        게임 영역 수동 설정 (창 내부의 특정 영역만 캡처하고 싶을 때)
        
        Args:
            x, y: 게임 영역 시작 좌표 (창 기준)
            width, height: 게임 영역 크기
        """
        if not self.window:
            logger.warning("창이 설정되지 않았습니다. find_window()를 먼저 호출하세요.")
            return
        
        self.game_area = {
            'left': self.window.left + x,
            'top': self.window.top + y,
            'width': width,
            'height': height
        }
        logger.info(f"게임 영역 설정: {self.game_area}")
    
    def capture(self) -> Optional[np.ndarray]:
        """
        현재 게임 화면 캡처
        
        Returns:
            캡처된 이미지 (numpy array, RGB 형식) 또는 None
        """
        if not self.game_area:
            logger.error("게임 영역이 설정되지 않았습니다.")
            return None
        
        try:
            with mss.mss() as sct:
                # 스크린샷 캡처
                screenshot = sct.grab(self.game_area)
                
                # numpy array로 변환 (BGRA → RGB)
                img = np.array(screenshot)
                img = img[:, :, :3]  # Alpha 채널 제거
                img = img[:, :, ::-1]  # BGR → RGB
                
                self.last_screenshot = img
                self.last_capture_time = time.time()
                
                return img
                
        except Exception as e:
            logger.error(f"화면 캡처 실패: {e}")
            return None
    
    def capture_pil(self) -> Optional[Image.Image]:
        """
        PIL Image 형식으로 캡처
        
        Returns:
            PIL Image 또는 None
        """
        img_array = self.capture()
        if img_array is None:
            return None
        
        return Image.fromarray(img_array)
    
    def save_screenshot(self, filepath: str) -> bool:
        """
        마지막 캡처 이미지 저장
        
        Args:
            filepath: 저장할 파일 경로
            
        Returns:
            성공 여부
        """
        if self.last_screenshot is None:
            logger.warning("저장할 스크린샷이 없습니다.")
            return False
        
        try:
            img = Image.fromarray(self.last_screenshot)
            img.save(filepath)
            logger.info(f"스크린샷 저장: {filepath}")
            return True
        except Exception as e:
            logger.error(f"스크린샷 저장 실패: {e}")
            return False
    
    def get_game_dimensions(self) -> Optional[Tuple[int, int]]:
        """
        게임 영역 크기 반환
        
        Returns:
            (width, height) 또는 None
        """
        if not self.game_area:
            return None
        return (self.game_area['width'], self.game_area['height'])
    
    def refresh_window_position(self) -> bool:
        """
        창 위치 갱신 (창이 이동했을 때)
        
        Returns:
            성공 여부
        """
        if not self.window:
            return False
        
        try:
            # 창 정보 업데이트
            windows = gw.getWindowsWithTitle(self.window_title_pattern)
            if not windows:
                return False
            
            self.window = windows[0]
            
            # 게임 영역 업데이트
            offset_x = self.game_area['left'] - self.window.left if self.game_area else 0
            offset_y = self.game_area['top'] - self.window.top if self.game_area else 0
            
            self.game_area = {
                'left': self.window.left + offset_x,
                'top': self.window.top + offset_y,
                'width': self.game_area['width'] if self.game_area else self.window.width,
                'height': self.game_area['height'] if self.game_area else self.window.height
            }
            
            return True
            
        except Exception as e:
            logger.error(f"창 위치 갱신 실패: {e}")
            return False


# 테스트 코드
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== 화면 캡처 테스트 ===\n")
    
    # ScreenCapture 인스턴스 생성
    capturer = ScreenCapture("MuMu")
    
    # 창 찾기
    if capturer.find_window():
        print("✅ MuMu Player 창을 찾았습니다.")
        
        # 게임 영역 크기 출력
        dimensions = capturer.get_game_dimensions()
        if dimensions:
            print(f"   게임 영역 크기: {dimensions[0]} x {dimensions[1]}")
        
        # 화면 캡처 테스트
        print("\n화면 캡처 중...")
        img = capturer.capture()
        
        if img is not None:
            print(f"✅ 캡처 성공: {img.shape}")
            
            # 스크린샷 저장
            test_path = "/home/ubuntu/coin_game_assistant/test_screenshot.png"
            if capturer.save_screenshot(test_path):
                print(f"✅ 스크린샷 저장: {test_path}")
        else:
            print("❌ 캡처 실패")
    else:
        print("❌ MuMu Player 창을 찾을 수 없습니다.")
        print("   MuMu Player가 실행 중인지 확인하세요.")
