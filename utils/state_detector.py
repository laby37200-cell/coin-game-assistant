"""
상태 감지 유틸리티

게임 화면의 정지 상태를 감지하여 API 호출 타이밍을 제어합니다.
"""

import time
import numpy as np
import logging
from typing import Optional
from collections import deque


logger = logging.getLogger(__name__)


class StateDetector:
    """게임 정지 상태 감지 클래스"""
    
    def __init__(
        self,
        check_frames: int = 5,
        pixel_threshold: int = 100,
        wait_time: float = 0.5
    ):
        """
        Args:
            check_frames: 연속으로 확인할 프레임 수
            pixel_threshold: 프레임 간 차이 픽셀 임계값
            wait_time: 안정 상태 확인 대기 시간 (초)
        """
        self.check_frames = check_frames
        self.pixel_threshold = pixel_threshold
        self.wait_time = wait_time
        
        # 최근 프레임 저장 (deque)
        self.recent_frames = deque(maxlen=check_frames)
        
        # 마지막 안정 상태 시간
        self.last_stable_time = 0
        
        logger.info(f"StateDetector 초기화: frames={check_frames}, "
                   f"threshold={pixel_threshold}, wait={wait_time}s")
    
    def add_frame(self, frame: np.ndarray):
        """
        새 프레임 추가
        
        Args:
            frame: 캡처된 프레임 (numpy array)
        """
        self.recent_frames.append(frame)
    
    def is_stable(self) -> bool:
        """
        현재 상태가 안정적인지 확인
        
        Returns:
            안정 상태 여부
        """
        # 충분한 프레임이 없으면 불안정
        if len(self.recent_frames) < self.check_frames:
            return False
        
        # 모든 연속 프레임 쌍을 비교
        for i in range(len(self.recent_frames) - 1):
            frame1 = self.recent_frames[i]
            frame2 = self.recent_frames[i + 1]
            
            # 프레임 차이 계산
            diff = self._calculate_frame_difference(frame1, frame2)
            
            # 임계값 초과 시 불안정
            if diff > self.pixel_threshold:
                return False
        
        # 모든 프레임이 유사하면 안정
        return True
    
    def _calculate_frame_difference(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray
    ) -> float:
        """
        두 프레임 간의 차이 계산
        
        Args:
            frame1, frame2: 비교할 프레임
            
        Returns:
            차이 값 (픽셀 수)
        """
        # 크기가 다르면 큰 차이로 간주
        if frame1.shape != frame2.shape:
            return float('inf')
        
        # 절대 차이 계산
        diff = np.abs(frame1.astype(float) - frame2.astype(float))
        
        # 차이가 있는 픽셀 수 계산 (임계값 10)
        diff_pixels = np.sum(diff > 10)
        
        return diff_pixels
    
    def wait_for_stability(
        self,
        capture_func,
        max_wait: float = 10.0,
        check_interval: float = 0.2
    ) -> bool:
        """
        안정 상태가 될 때까지 대기
        
        Args:
            capture_func: 프레임 캡처 함수 (인자 없음, numpy array 반환)
            max_wait: 최대 대기 시간 (초)
            check_interval: 확인 간격 (초)
            
        Returns:
            안정 상태 도달 여부
        """
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            # 프레임 캡처
            frame = capture_func()
            
            if frame is None:
                logger.warning("프레임 캡처 실패")
                time.sleep(check_interval)
                continue
            
            # 프레임 추가
            self.add_frame(frame)
            
            # 안정 상태 확인
            if self.is_stable():
                # 추가 대기 시간
                time.sleep(self.wait_time)
                
                # 다시 확인
                frame = capture_func()
                if frame is not None:
                    self.add_frame(frame)
                    
                    if self.is_stable():
                        logger.info("안정 상태 도달")
                        self.last_stable_time = time.time()
                        return True
            
            # 대기
            time.sleep(check_interval)
        
        logger.warning(f"안정 상태 대기 시간 초과 ({max_wait}s)")
        return False
    
    def reset(self):
        """상태 초기화"""
        self.recent_frames.clear()
        logger.debug("StateDetector 리셋")
    
    def get_stability_score(self) -> float:
        """
        현재 안정도 점수 반환 (0~1, 1이 가장 안정)
        
        Returns:
            안정도 점수
        """
        if len(self.recent_frames) < 2:
            return 0.0
        
        total_diff = 0
        count = 0
        
        for i in range(len(self.recent_frames) - 1):
            frame1 = self.recent_frames[i]
            frame2 = self.recent_frames[i + 1]
            
            diff = self._calculate_frame_difference(frame1, frame2)
            total_diff += diff
            count += 1
        
        avg_diff = total_diff / count if count > 0 else float('inf')
        
        # 점수 계산 (임계값 대비)
        score = max(0, 1 - (avg_diff / self.pixel_threshold))
        
        return score


# 테스트 코드
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== 상태 감지 테스트 ===\n")
    
    # StateDetector 생성
    detector = StateDetector(check_frames=3, pixel_threshold=100)
    
    # 테스트 프레임 생성
    print("테스트 프레임 생성 중...")
    
    # 정적 프레임 (안정)
    static_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # 동적 프레임 (불안정)
    dynamic_frames = [
        np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        for _ in range(5)
    ]
    
    # 테스트 1: 정적 프레임
    print("\n[테스트 1: 정적 프레임]")
    detector.reset()
    
    for i in range(5):
        detector.add_frame(static_frame)
        stable = detector.is_stable()
        score = detector.get_stability_score()
        print(f"  프레임 {i+1}: 안정={stable}, 점수={score:.2f}")
    
    # 테스트 2: 동적 프레임
    print("\n[테스트 2: 동적 프레임]")
    detector.reset()
    
    for i, frame in enumerate(dynamic_frames):
        detector.add_frame(frame)
        stable = detector.is_stable()
        score = detector.get_stability_score()
        print(f"  프레임 {i+1}: 안정={stable}, 점수={score:.2f}")
    
    print("\n✅ 테스트 완료")
