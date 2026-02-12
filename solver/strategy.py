"""
전략 평가 모듈

수박게임 공략 전략을 기반으로 게임 상태를 평가합니다.
"""

import logging
from typing import List, Dict, Tuple
from models.coin import Coin, CoinType


logger = logging.getLogger(__name__)


class StrategyEvaluator:
    """수박게임 전략 기반 상태 평가 클래스"""
    
    # 천장 라인 — 이 위로 동전이 올라가면 게임오버
    # 이미지 기준 y좌표 (screen coords, y-down). Gemini가 감지한 좌표 기준.
    DEFAULT_CEILING_Y = 200  # 흰 점선 위치 (화면 상단에서 약 200px)

    def __init__(
        self,
        game_width: int,
        game_height: int,
        ceiling_y: float = None,
        weight_large_coin: float = 10.0,
        weight_adjacency: float = 8.0,
        weight_height_penalty: float = -3.0,
        weight_corner_bonus: float = 3.0,
        weight_blocking_penalty: float = -20.0,
        weight_chain_merge: float = 30.0,
        prefer_corner: bool = True,
        left_to_right: bool = True
    ):
        """
        Args:
            game_width: 게임 영역 너비
            game_height: 게임 영역 높이
            ceiling_y: 천장 y좌표 (screen coords). 이 위로 동전이 올라가면 게임오버.
            weight_large_coin: 큰 동전 보너스 가중치
            weight_adjacency: 인접도 보너스 가중치
            weight_height_penalty: 높이 페널티 가중치
            weight_corner_bonus: 구석 배치 보너스 가중치
            weight_blocking_penalty: 블로킹 페널티 가중치
            weight_chain_merge: 연쇄 합체 보너스 가중치
            prefer_corner: 큰 동전을 구석에 배치하는 전략 사용
            left_to_right: 좌→우 정렬 전략 사용
        """
        self.game_width = game_width
        self.game_height = game_height
        self.ceiling_y = ceiling_y if ceiling_y is not None else self.DEFAULT_CEILING_Y
        
        # 가중치
        self.weight_large_coin = weight_large_coin
        self.weight_adjacency = weight_adjacency
        self.weight_height_penalty = weight_height_penalty
        self.weight_corner_bonus = weight_corner_bonus
        self.weight_blocking_penalty = weight_blocking_penalty
        self.weight_chain_merge = weight_chain_merge
        
        # 전략 플래그
        self.prefer_corner = prefer_corner
        self.left_to_right = left_to_right
        
        logger.info(f"StrategyEvaluator 초기화: ceiling_y={self.ceiling_y}, target=2700")
    
    def evaluate(self, coins: List[Coin]) -> float:
        """
        게임 상태 종합 평가
        
        Args:
            coins: 현재 게임 상태의 동전 리스트
            
        Returns:
            평가 점수 (높을수록 좋음)
        """
        if not coins:
            return 0.0
        
        score = 0.0
        
        # 0. 천장 게임오버 체크 (최우선 — 천장 넘으면 치명적 페널티)
        score += self._evaluate_ceiling(coins)
        
        # 1. 큰 동전 보너스
        score += self._evaluate_large_coins(coins)
        
        # 2. 같은 동전 인접도 보너스 (합체 가능성)
        score += self._evaluate_adjacency(coins)
        
        # 3. 높이 페널티 (천장에 가까울수록 지수적 증가)
        score += self._evaluate_height(coins)
        
        # 4. 구석 배치 보너스
        if self.prefer_corner:
            score += self._evaluate_corner_placement(coins)
        
        # 5. 블로킹 페널티
        score += self._evaluate_blocking(coins)
        
        # 6. 좌우 정렬 보너스
        if self.left_to_right:
            score += self._evaluate_sorting(coins)
        
        # 7. 연쇄 합체 보너스
        score += self._evaluate_chain_merge(coins)
        
        # 8. 동전 수 감소 보너스 (동전이 적을수록 좋음 — 합체가 잘 된 상태)
        score -= len(coins) * 2.0
        
        return score
    
    def _evaluate_large_coins(self, coins: List[Coin]) -> float:
        """큰 동전 보너스 계산"""
        score = 0.0
        
        for coin in coins:
            # 레벨이 높을수록 높은 점수
            score += coin.coin_type.level * self.weight_large_coin
        
        return score
    
    def _evaluate_adjacency(self, coins: List[Coin]) -> float:
        """같은 동전 인접도 보너스 계산"""
        score = 0.0
        
        for i, coin1 in enumerate(coins):
            for coin2 in coins[i+1:]:
                # 같은 종류의 동전만 평가
                if coin1.coin_type != coin2.coin_type:
                    continue
                
                distance = coin1.distance_to(coin2)
                
                # 합체 가능 거리 (반지름 합의 2배 이내)
                merge_distance = (coin1.radius + coin2.radius) * 2
                
                if distance < merge_distance:
                    # 가까울수록 높은 점수
                    adjacency_bonus = self.weight_adjacency * (merge_distance - distance) / merge_distance
                    score += adjacency_bonus
        
        return score
    
    def _evaluate_ceiling(self, coins: List[Coin]) -> float:
        """천장 게임오버 체크 — 동전이 천장 위로 올라가면 치명적 페널티"""
        penalty = 0.0
        for coin in coins:
            coin_top = coin.y - coin.radius  # screen coords (y-down)
            if coin_top < self.ceiling_y:
                # 천장 위로 올라간 동전 — 게임오버 위험
                over = self.ceiling_y - coin_top
                penalty -= 5000 + over * 100  # 치명적 페널티
        return penalty

    def _evaluate_height(self, coins: List[Coin]) -> float:
        """높이 페널티 계산 — 천장에 가까울수록 지수적으로 증가"""
        if not coins:
            return 0.0
        
        penalty = 0.0
        for coin in coins:
            coin_top = coin.y - coin.radius  # screen coords (y-down)
            # 천장까지의 여유 거리
            margin = coin_top - self.ceiling_y
            if margin < 0:
                continue  # _evaluate_ceiling에서 처리
            elif margin < 50:
                # 천장 근처 — 지수적 페널티
                penalty += self.weight_height_penalty * (50 - margin) ** 2 * 0.1
            elif margin < 150:
                # 중간 높이 — 선형 페널티
                penalty += self.weight_height_penalty * (150 - margin) * 0.5
        
        return penalty
    
    def _evaluate_corner_placement(self, coins: List[Coin]) -> float:
        """구석 배치 보너스 계산"""
        score = 0.0
        
        # 큰 동전만 평가 (레벨 7 이상)
        large_coins = [c for c in coins if c.coin_type.level >= 7]
        
        corner_threshold = self.game_width * 0.2  # 양쪽 20% 영역
        
        for coin in large_coins:
            # 왼쪽 구석
            if coin.x < corner_threshold:
                score += self.weight_corner_bonus * coin.coin_type.level
            # 오른쪽 구석
            elif coin.x > self.game_width - corner_threshold:
                score += self.weight_corner_bonus * coin.coin_type.level
        
        return score
    
    def _evaluate_blocking(self, coins: List[Coin]) -> float:
        """블로킹 페널티 계산 (작은 동전이 큰 동전 사이에 끼인 경우)"""
        score = 0.0
        
        for coin in coins:
            # 작은 동전만 평가 (레벨 4 이하)
            if coin.coin_type.level > 4:
                continue
            
            # 주변에 큰 동전이 있는지 확인
            nearby_large_coins = []
            
            for other in coins:
                if other.coin_type.level <= 4:
                    continue
                
                distance = coin.distance_to(other)
                
                # 근처에 있는 큰 동전
                if distance < (coin.radius + other.radius) * 3:
                    nearby_large_coins.append(other)
            
            # 양쪽에 큰 동전이 있으면 블로킹으로 간주
            if len(nearby_large_coins) >= 2:
                # 좌우에 있는지 확인
                left_count = sum(1 for c in nearby_large_coins if c.x < coin.x)
                right_count = sum(1 for c in nearby_large_coins if c.x > coin.x)
                
                if left_count > 0 and right_count > 0:
                    # 블로킹 페널티
                    score += self.weight_blocking_penalty
        
        return score
    
    def _evaluate_sorting(self, coins: List[Coin]) -> float:
        """좌우 정렬 보너스 계산 (작은 동전 왼쪽, 큰 동전 오른쪽)"""
        score = 0.0
        
        if len(coins) < 2:
            return 0.0
        
        # 동전을 x 좌표로 정렬
        sorted_coins = sorted(coins, key=lambda c: c.x)
        
        # 레벨이 증가하는 경향이 있으면 보너스
        for i in range(len(sorted_coins) - 1):
            level_diff = sorted_coins[i+1].coin_type.level - sorted_coins[i].coin_type.level
            
            if level_diff > 0:
                score += 2.0 * level_diff
            elif level_diff < 0:
                score -= 0.5 * abs(level_diff)
        
        return score

    def _evaluate_chain_merge(self, coins: List[Coin]) -> float:
        """
        연쇄 합체 가능성 보너스.
        같은 종류 동전이 3개 이상 근접해 있으면 연쇄 합체 가능 → 큰 보너스.
        합체 후 생기는 상위 동전이 또 같은 종류와 인접하면 추가 보너스.
        """
        score = 0.0
        
        # 종류별로 동전 그룹핑
        from collections import defaultdict
        groups = defaultdict(list)
        for coin in coins:
            groups[coin.coin_type].append(coin)
        
        for coin_type, group in groups.items():
            if len(group) < 2:
                continue
            
            # 같은 종류 동전 쌍 중 합체 가능 거리에 있는 쌍 수
            merge_pairs = 0
            for i, c1 in enumerate(group):
                for c2 in group[i+1:]:
                    dist = c1.distance_to(c2)
                    touch_dist = (c1.radius + c2.radius) * 1.2
                    if dist < touch_dist:
                        merge_pairs += 1
            
            if merge_pairs >= 1:
                # 합체 가능 쌍이 있음
                score += self.weight_chain_merge * merge_pairs
                
                # 연쇄 가능성: 합체 후 상위 동전이 같은 종류와 인접한지
                next_type = coin_type.get_next_level()
                if next_type and next_type in groups:
                    for upper in groups[next_type]:
                        for c in group:
                            dist = c.distance_to(upper)
                            if dist < (c.radius + upper.radius) * 2.5:
                                # 연쇄 합체 가능!
                                score += self.weight_chain_merge * 2
                                break
        
        return score
    
    def get_evaluation_details(self, coins: List[Coin]) -> Dict[str, float]:
        """
        평가 상세 정보 반환 (디버깅용)
        
        Args:
            coins: 동전 리스트
            
        Returns:
            각 평가 항목별 점수
        """
        details = {
            'ceiling': self._evaluate_ceiling(coins),
            'large_coins': self._evaluate_large_coins(coins),
            'adjacency': self._evaluate_adjacency(coins),
            'height': self._evaluate_height(coins),
            'corner': self._evaluate_corner_placement(coins) if self.prefer_corner else 0.0,
            'blocking': self._evaluate_blocking(coins),
            'sorting': self._evaluate_sorting(coins) if self.left_to_right else 0.0,
            'chain_merge': self._evaluate_chain_merge(coins),
            'coin_count_penalty': -len(coins) * 2.0,
            'total': self.evaluate(coins)
        }
        
        return details


# 테스트 코드
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== 전략 평가 테스트 ===\n")
    
    # 평가자 생성
    evaluator = StrategyEvaluator(game_width=600, game_height=800)
    
    # 테스트 케이스 1: 좋은 배치
    good_coins = [
        Coin(CoinType.BLACK_THUNDER, x=100, y=750),
        Coin(CoinType.BLACK_THUNDER, x=130, y=750),  # 같은 동전 인접
        Coin(CoinType.YELLOW_CIRCLE, x=200, y=750),
        Coin(CoinType.MINT_GIFTBOX, x=500, y=750),  # 큰 동전 구석
    ]
    
    print("[좋은 배치]")
    score = evaluator.evaluate(good_coins)
    details = evaluator.get_evaluation_details(good_coins)
    
    print(f"총점: {score:.1f}")
    for key, value in details.items():
        if key != 'total':
            print(f"  {key}: {value:.1f}")
    
    # 테스트 케이스 2: 나쁜 배치
    bad_coins = [
        Coin(CoinType.MINT_GIFTBOX, x=100, y=100),  # 큰 동전이 높이 쌓임
        Coin(CoinType.BLACK_THUNDER, x=300, y=750),
        Coin(CoinType.BLUE_CIRCLE, x=320, y=750),  # 작은 동전이 큰 동전 사이
        Coin(CoinType.BLUE_CIRCLE, x=350, y=750),
    ]
    
    print("\n[나쁜 배치]")
    score = evaluator.evaluate(bad_coins)
    details = evaluator.get_evaluation_details(bad_coins)
    
    print(f"총점: {score:.1f}")
    for key, value in details.items():
        if key != 'total':
            print(f"  {key}: {value:.1f}")
    
    print("\n✅ 테스트 완료")
