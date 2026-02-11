"""
좌표 변환 유틸리티

화면 좌표계와 물리 엔진 좌표계 간의 변환을 처리합니다.
"""

import logging
from typing import Tuple


logger = logging.getLogger(__name__)


class CoordinateMapper:
    """좌표계 변환 클래스"""
    
    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        physics_width: int,
        physics_height: int,
        scale: float = 1.0
    ):
        """
        Args:
            screen_width: 화면 좌표계 너비
            screen_height: 화면 좌표계 높이
            physics_width: 물리 엔진 좌표계 너비
            physics_height: 물리 엔진 좌표계 높이
            scale: 스케일 팩터
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.physics_width = physics_width
        self.physics_height = physics_height
        self.scale = scale
        
        logger.info(f"CoordinateMapper 초기화: "
                   f"화면({screen_width}x{screen_height}) → "
                   f"물리({physics_width}x{physics_height})")
    
    def screen_to_physics(self, x: float, y: float) -> Tuple[float, float]:
        """
        화면 좌표 → 물리 엔진 좌표 변환
        
        화면 좌표계: 왼쪽 상단이 (0, 0), y축 아래 방향이 양수
        물리 좌표계: 왼쪽 하단이 (0, 0), y축 위 방향이 양수 (일반적인 수학 좌표계)
        
        Args:
            x, y: 화면 좌표
            
        Returns:
            (physics_x, physics_y): 물리 엔진 좌표
        """
        # x는 동일 (왼쪽이 0)
        physics_x = x * self.scale
        
        # y는 반전 (상단 0 → 하단 0)
        physics_y = self.physics_height - (y * self.scale)
        
        return physics_x, physics_y
    
    def physics_to_screen(self, x: float, y: float) -> Tuple[float, float]:
        """
        물리 엔진 좌표 → 화면 좌표 변환
        
        Args:
            x, y: 물리 엔진 좌표
            
        Returns:
            (screen_x, screen_y): 화면 좌표
        """
        # x는 동일
        screen_x = x / self.scale
        
        # y는 반전
        screen_y = (self.physics_height - y) / self.scale
        
        return screen_x, screen_y
    
    def screen_to_physics_distance(self, distance: float) -> float:
        """
        화면 거리 → 물리 엔진 거리 변환
        
        Args:
            distance: 화면 거리
            
        Returns:
            물리 엔진 거리
        """
        return distance * self.scale
    
    def physics_to_screen_distance(self, distance: float) -> float:
        """
        물리 엔진 거리 → 화면 거리 변환
        
        Args:
            distance: 물리 엔진 거리
            
        Returns:
            화면 거리
        """
        return distance / self.scale


# 테스트 코드
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== 좌표 변환 테스트 ===\n")
    
    # CoordinateMapper 생성
    mapper = CoordinateMapper(
        screen_width=600,
        screen_height=800,
        physics_width=600,
        physics_height=800
    )
    
    # 테스트 케이스
    test_cases = [
        (0, 0, "왼쪽 상단"),
        (600, 0, "오른쪽 상단"),
        (0, 800, "왼쪽 하단"),
        (600, 800, "오른쪽 하단"),
        (300, 400, "중앙"),
    ]
    
    print("화면 좌표 → 물리 좌표 → 화면 좌표:")
    for screen_x, screen_y, label in test_cases:
        # 화면 → 물리
        physics_x, physics_y = mapper.screen_to_physics(screen_x, screen_y)
        
        # 물리 → 화면 (역변환)
        back_x, back_y = mapper.physics_to_screen(physics_x, physics_y)
        
        print(f"  {label}: ({screen_x}, {screen_y}) → "
              f"({physics_x:.1f}, {physics_y:.1f}) → "
              f"({back_x:.1f}, {back_y:.1f})")
        
        # 검증
        assert abs(back_x - screen_x) < 0.01, "x 좌표 변환 오류"
        assert abs(back_y - screen_y) < 0.01, "y 좌표 변환 오류"
    
    print("\n✅ 모든 변환이 정확합니다.")
