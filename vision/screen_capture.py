"""
화면 캡처 모듈

MuMu Player 창을 찾아서 게임 화면을 실시간으로 캡처합니다.
"""

import time
import numpy as np
from PIL import Image
from typing import Optional, Tuple
import logging
import ctypes

try:
    import mss
except ImportError:
    mss = None

try:
    import pygetwindow as gw
except ImportError:
    gw = None

try:
    import win32gui
    import win32ui
    import win32con
    _HAS_WIN32 = True
except ImportError:
    _HAS_WIN32 = False


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
    
    # 브라우저 등 에뮬레이터가 아닌 창을 걸러내기 위한 키워드
    _EXCLUDE_KEYWORDS = ("Chrome", "Edge", "Firefox", "Opera", "Brave", "Safari")

    def find_window(self) -> bool:
        """
        MuMu Player 창 찾기
        
        Returns:
            성공 여부
        """
        try:
            # 여러 패턴으로 시도 (Android Device 우선 — 실제 게임 렌더링 창)
            patterns = [self.window_title_pattern]
            if "MuMu" in self.window_title_pattern:
                patterns = ["Android Device", "MuMu Android Device",
                            self.window_title_pattern, "MuMuPlayer", "MuMu Player"]

            candidates = []
            seen_titles = set()
            for pat in patterns:
                for w in gw.getWindowsWithTitle(pat):
                    if w.title not in seen_titles:
                        seen_titles.add(w.title)
                        candidates.append(w)

            if not candidates:
                logger.warning(f"'{self.window_title_pattern}' 패턴을 가진 창을 찾을 수 없습니다.")
                return False

            # 브라우저 창 제외
            filtered = [w for w in candidates
                        if not any(kw in w.title for kw in self._EXCLUDE_KEYWORDS)]
            if not filtered:
                filtered = candidates  # 모두 브라우저면 원본 사용

            # 가장 적합한 창 선택 (크기가 유효한 것 우선)
            valid = [w for w in filtered if w.width > 100 and w.height > 100]
            self.window = valid[0] if valid else filtered[0]

            logger.info(f"창 발견: '{self.window.title}' at ({self.window.left}, {self.window.top}) "
                        f"size={self.window.width}x{self.window.height}")
            
            # 게임 영역 설정 — 클라이언트 영역 기준 (타이틀바/테두리 제외)
            area_left = self.window.left
            area_top = self.window.top
            area_w = self.window.width
            area_h = self.window.height

            if _HAS_WIN32:
                try:
                    hwnd = self._get_hwnd()
                    if hwnd:
                        # 클라이언트 영역의 화면 좌표
                        pt = win32gui.ClientToScreen(hwnd, (0, 0))
                        cr = win32gui.GetClientRect(hwnd)
                        area_left = pt[0]
                        area_top = pt[1]
                        area_w = cr[2] - cr[0]
                        area_h = cr[3] - cr[1]
                        logger.info(f"클라이언트 영역: ({area_left},{area_top}) {area_w}x{area_h}")
                except Exception as e:
                    logger.debug(f"ClientToScreen 실패: {e}")

            self.game_area = {
                'left': area_left,
                'top': area_top,
                'width': area_w,
                'height': area_h
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
        
        # Win32 API 캡처 우선 (GPU 렌더링 에뮬레이터 대응)
        if _HAS_WIN32 and self.window:
            img = self._capture_win32()
            if img is not None:
                return img
        
        # Fallback: mss
        return self._capture_mss()

    def _capture_win32(self) -> Optional[np.ndarray]:
        """Win32 PrintWindow/BitBlt 기반 캡처 (GPU 렌더링 대응)"""
        try:
            hwnd = self._get_hwnd()
            if not hwnd:
                return None

            left, top, right, bottom = win32gui.GetClientRect(hwnd)
            w = right - left
            h = bottom - top
            if w <= 0 or h <= 0:
                return None

            hwndDC = win32gui.GetWindowDC(hwnd)
            mfcDC = win32ui.CreateDCFromHandle(hwndDC)
            saveDC = mfcDC.CreateCompatibleDC()

            saveBitMap = win32ui.CreateBitmap()
            saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
            saveDC.SelectObject(saveBitMap)

            # PrintWindow with PW_RENDERFULLCONTENT (flag=2) for GPU content
            ctypes.windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 2)

            bmpinfo = saveBitMap.GetInfo()
            bmpstr = saveBitMap.GetBitmapBits(True)

            img = np.frombuffer(bmpstr, dtype=np.uint8)
            img = img.reshape((bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4))
            img = img[:, :, :3]  # BGRA → BGR
            img = img[:, :, ::-1]  # BGR → RGB

            # 리소스 해제
            win32gui.DeleteObject(saveBitMap.GetHandle())
            saveDC.DeleteDC()
            mfcDC.DeleteDC()
            win32gui.ReleaseDC(hwnd, hwndDC)

            # 검은 화면 체크
            if img.mean() < 5:
                logger.debug("Win32 캡처 결과가 검은 화면 — fallback")
                return None

            self.last_screenshot = img
            self.last_capture_time = time.time()

            # game_area를 클라이언트 영역 기준으로 동기화
            # ClientToScreen으로 클라이언트 영역의 실제 화면 좌표를 구함
            try:
                pt = win32gui.ClientToScreen(hwnd, (0, 0))
                self.game_area['left'] = pt[0]
                self.game_area['top'] = pt[1]
            except Exception:
                pass  # fallback: 기존 window.left/top 유지
            self.game_area['width'] = w
            self.game_area['height'] = h

            return img

        except Exception as e:
            logger.debug(f"Win32 캡처 실패: {e}")
            return None

    def _get_hwnd(self) -> Optional[int]:
        """pygetwindow 창 객체에서 Win32 HWND 핸들 추출"""
        try:
            if hasattr(self.window, '_hWnd'):
                return self.window._hWnd
            # fallback: FindWindow
            hwnd = win32gui.FindWindow(None, self.window.title)
            return hwnd if hwnd else None
        except Exception:
            return None

    def _capture_mss(self) -> Optional[np.ndarray]:
        """mss 기반 캡처 (fallback)"""
        if mss is None:
            return None
        try:
            with mss.mss() as sct:
                screenshot = sct.grab(self.game_area)
                img = np.array(screenshot)
                img = img[:, :, :3]
                img = img[:, :, ::-1]

                self.last_screenshot = img
                self.last_capture_time = time.time()
                return img
        except Exception as e:
            logger.error(f"mss 캡처 실패: {e}")
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
