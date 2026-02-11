"""
콘텐츠페이 동전게임 동전 계층 구조 정의

실제 게임 이미지를 분석하여 정확한 동전 크기와 색상을 반영합니다.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple


class CoinType(Enum):
    """동전 종류 정의 (작은 것부터 큰 순서)"""
    
    # 레벨 1-11: 콘텐츠페이 동전게임 계층
    # 실제 게임 이미지 분석 기반 (425x189px 화면)
    BLACK_THUNDER = (1, "검정번개", 18, 10, (50, 50, 50))           # 가장 작음
    PINK_CIRCLE = (2, "핑크동전", 27, 20, (255, 150, 180))
    ORANGE_CIRCLE = (3, "주황동전", 35, 40, (255, 140, 100))
    YELLOW_CIRCLE = (4, "노랑동전", 42, 80, (255, 220, 100))
    MINT_CIRCLE = (5, "민트동전", 48, 160, (150, 255, 200))
    BLUE_CIRCLE = (6, "파랑동전", 55, 320, (100, 180, 255))        # 랜덤 최대
    PURPLE_CIRCLE = (7, "보라동전", 62, 640, (180, 120, 255))
    BROWN_CIRCLE = (8, "갈색동전", 70, 1280, (150, 100, 70))
    WHITE_SQUARE = (9, "흰색상자", 78, 2560, (240, 240, 240))
    YELLOW_BULB = (10, "노랑전구", 87, 5120, (255, 230, 100))
    MINT_GIFTBOX = (11, "민트선물상자", 96, 10240, (120, 255, 200))  # 가장 큼, 최종 목표
    
    def __init__(self, level: int, name: str, radius: int, score: int, color: Tuple[int, int, int]):
        self.level = level
        self.display_name = name
        self.radius = radius  # 픽셀 단위 반지름
        self.score = score    # 합체 시 획득 점수
        self.color = color    # RGB 색상
    
    @property
    def mass(self) -> float:
        """질량 계산 (반지름의 제곱에 비례)"""
        return (self.radius ** 2) * 0.01
    
    @property
    def diameter(self) -> int:
        """지름"""
        return self.radius * 2
    
    def can_merge_with(self, other: 'CoinType') -> bool:
        """다른 동전과 합체 가능 여부"""
        return self.level == other.level
    
    def get_next_level(self) -> Optional['CoinType']:
        """합체 후 생성되는 다음 레벨 동전"""
        next_level = self.level + 1
        for coin in CoinType:
            if coin.level == next_level:
                return coin
        return None  # 최고 레벨인 경우 (민트 선물상자)
    
    @classmethod
    def from_name(cls, name: str) -> Optional['CoinType']:
        """이름으로 동전 타입 찾기"""
        # 다양한 이름 패턴 지원
        name_lower = name.lower().replace(" ", "")
        
        for coin in cls:
            coin_name_lower = coin.display_name.lower().replace(" ", "")
            if coin_name_lower in name_lower or name_lower in coin_name_lower:
                return coin
            if coin.name.lower() == name_lower:
                return coin
        
        return None
    
    @classmethod
    def from_level(cls, level: int) -> Optional['CoinType']:
        """레벨로 동전 타입 찾기"""
        for coin in cls:
            if coin.level == level:
                return coin
        return None
    
    @classmethod
    def get_random_drop_coins(cls) -> list['CoinType']:
        """랜덤으로 떨어질 수 있는 동전 리스트 (레벨 1~6)"""
        return [coin for coin in cls if coin.level <= 6]
    
    @classmethod
    def is_random_droppable(cls, coin_type: 'CoinType') -> bool:
        """랜덤으로 떨어질 수 있는 동전인지 확인"""
        return coin_type.level <= 6


@dataclass
class Coin:
    """게임 내 동전 인스턴스"""
    
    coin_type: CoinType
    x: float  # 중심 x 좌표 (픽셀)
    y: float  # 중심 y 좌표 (픽셀)
    velocity_x: float = 0.0  # x 방향 속도
    velocity_y: float = 0.0  # y 방향 속도
    
    @property
    def radius(self) -> int:
        """반지름"""
        return self.coin_type.radius
    
    @property
    def mass(self) -> float:
        """질량"""
        return self.coin_type.mass
    
    @property
    def position(self) -> Tuple[float, float]:
        """위치 튜플"""
        return (self.x, self.y)
    
    def distance_to(self, other: 'Coin') -> float:
        """다른 동전과의 거리"""
        import math
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    
    def is_touching(self, other: 'Coin') -> bool:
        """다른 동전과 접촉 여부"""
        return self.distance_to(other) <= (self.radius + other.radius)
    
    def can_merge_with(self, other: 'Coin') -> bool:
        """다른 동전과 합체 가능 여부"""
        return self.coin_type.can_merge_with(other.coin_type) and self.is_touching(other)
    
    def to_dict(self) -> dict:
        """딕셔너리로 변환 (JSON 직렬화용)"""
        return {
            'type': self.coin_type.display_name,
            'level': self.coin_type.level,
            'x': self.x,
            'y': self.y,
            'radius': self.radius,
            'velocity_x': self.velocity_x,
            'velocity_y': self.velocity_y
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Coin':
        """딕셔너리로부터 생성"""
        coin_type = CoinType.from_name(data['type']) or CoinType.from_level(data['level'])
        if not coin_type:
            raise ValueError(f"Unknown coin type: {data['type']}")
        
        return cls(
            coin_type=coin_type,
            x=data['x'],
            y=data['y'],
            velocity_x=data.get('velocity_x', 0.0),
            velocity_y=data.get('velocity_y', 0.0)
        )


# 동전 계층 구조 정보 출력 (디버깅용)
if __name__ == "__main__":
    print("=== 콘텐츠페이 동전게임 계층 구조 ===\n")
    
    for coin in CoinType:
        print(f"레벨 {coin.level:2d}: {coin.display_name:12s} | "
              f"반지름: {coin.radius:3d}px | "
              f"질량: {coin.mass:6.2f} | "
              f"점수: {coin.score:5d} | "
              f"색상: RGB{coin.color}")
        
        next_coin = coin.get_next_level()
        if next_coin:
            print(f"         → 합체 시: {next_coin.display_name}")
        else:
            print(f"         → 최고 레벨 (합체 시 사라짐)")
        
        # 랜덤 드롭 가능 여부
        if CoinType.is_random_droppable(coin):
            print(f"         ✓ 랜덤으로 떨어질 수 있음")
        
        print()
    
    print("\n=== 랜덤 드롭 가능 동전 ===")
    random_coins = CoinType.get_random_drop_coins()
    for coin in random_coins:
        print(f"  - {coin.display_name} (레벨 {coin.level})")
