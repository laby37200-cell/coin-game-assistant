"""
물리 시뮬레이터

현재 게임 상태를 복제하여 Digital Twin을 생성하고,
특정 위치에 동전을 떨어뜨렸을 때의 결과를 시뮬레이션합니다.
"""

import copy
import logging
from typing import List, Tuple, Optional

from models.coin import Coin, CoinType
from physics.engine import PhysicsEngine


logger = logging.getLogger(__name__)


class PhysicsSimulator:
    """물리 시뮬레이션 실행 클래스"""
    
    def __init__(
        self,
        game_width: int,
        game_height: int,
        time_step: float = 1/60,
        simulation_duration: float = 3.0
    ):
        """
        Args:
            game_width: 게임 영역 너비
            game_height: 게임 영역 높이
            time_step: 시뮬레이션 타임스텝 (초)
            simulation_duration: 시뮬레이션 지속 시간 (초)
        """
        self.game_width = game_width
        self.game_height = game_height
        self.time_step = time_step
        self.simulation_duration = simulation_duration
        
        # 물리 엔진 (재사용)
        self.engine = PhysicsEngine(game_width, game_height)
        
        logger.info(f"PhysicsSimulator 초기화: {game_width}x{game_height}")
    
    def create_digital_twin(self, coins: List[Coin]) -> PhysicsEngine:
        """
        현재 게임 상태를 물리 공간에 복제 (Digital Twin)
        
        Args:
            coins: 현재 게임에 있는 동전 리스트
            
        Returns:
            복제된 PhysicsEngine
        """
        # 엔진 초기화
        self.engine.clear()
        
        # 모든 동전을 물리 공간에 추가 (정적 바디로)
        for coin in coins:
            self.engine.add_coin(coin, is_static=True)
        
        logger.debug(f"Digital Twin 생성: 동전 {len(coins)}개 복제")
        
        return self.engine
    
    def simulate_drop(
        self,
        current_coins: List[Coin],
        drop_coin_type: CoinType,
        drop_x: float,
        drop_y: float = 50
    ) -> Tuple[List[Coin], float]:
        """
        특정 위치에 동전을 떨어뜨렸을 때의 결과 시뮬레이션
        
        Args:
            current_coins: 현재 게임 상태의 동전 리스트
            drop_coin_type: 떨어뜨릴 동전 종류
            drop_x: 떨어뜨릴 x 좌표
            drop_y: 떨어뜨릴 y 좌표 (기본값: 상단)
            
        Returns:
            (시뮬레이션 후 동전 리스트, 예상 점수)
        """
        # Digital Twin 생성
        self.create_digital_twin(current_coins)
        
        # 떨어뜨릴 동전 생성
        drop_coin = Coin(
            coin_type=drop_coin_type,
            x=drop_x,
            y=drop_y
        )
        
        # 동적 바디로 추가
        drop_coin_id = self.engine.add_coin(drop_coin, is_static=False)
        
        # 시뮬레이션 실행
        total_steps = int(self.simulation_duration / self.time_step)
        
        for step in range(total_steps):
            self.engine.step(self.time_step)
            
            # 조기 종료: 안정 상태 도달
            if step > 60 and self.engine.is_stable():  # 최소 1초 후
                logger.debug(f"안정 상태 도달: step={step}")
                break
        
        # 최종 상태 추출
        final_coins = self.engine.get_all_coins_state()
        
        # 점수 계산 (간단한 버전)
        score = self._calculate_score(final_coins)
        
        logger.debug(f"시뮬레이션 완료: x={drop_x:.1f}, 점수={score:.1f}")
        
        return final_coins, score
    
    def _calculate_score(self, coins: List[Coin]) -> float:
        """
        동전 배치 상태 평가 점수 계산
        
        Args:
            coins: 동전 리스트
            
        Returns:
            평가 점수 (높을수록 좋음)
        """
        score = 0.0
        
        # 1. 큰 동전 보너스
        for coin in coins:
            score += coin.coin_type.level * 10
        
        # 2. 같은 동전 인접도 보너스
        for i, coin1 in enumerate(coins):
            for coin2 in coins[i+1:]:
                if coin1.coin_type == coin2.coin_type:
                    distance = coin1.distance_to(coin2)
                    # 가까울수록 높은 점수
                    if distance < (coin1.radius + coin2.radius) * 2:
                        adjacency_bonus = 50 / (distance + 1)
                        score += adjacency_bonus
        
        # 3. 높이 페널티 (높이 쌓일수록 감점)
        max_height = 0
        for coin in coins:
            coin_top = coin.y - coin.radius
            if coin_top < max_height or max_height == 0:
                max_height = coin_top
        
        height_penalty = max_height * 0.1
        score -= height_penalty
        
        # 4. 구석 배치 보너스 (큰 동전이 구석에 있으면 보너스)
        for coin in coins:
            if coin.coin_type.level >= 7:  # 큰 동전만
                # 왼쪽 또는 오른쪽 구석
                if coin.x < 100 or coin.x > self.game_width - 100:
                    score += 30
        
        return score
    
    def simulate_multiple_positions(
        self,
        current_coins: List[Coin],
        drop_coin_type: CoinType,
        x_positions: List[float],
        drop_y: float = 50
    ) -> List[Tuple[float, float, List[Coin]]]:
        """
        여러 x 위치에 대해 시뮬레이션 실행
        
        Args:
            current_coins: 현재 게임 상태
            drop_coin_type: 떨어뜨릴 동전 종류
            x_positions: 테스트할 x 좌표 리스트
            drop_y: 떨어뜨릴 y 좌표
            
        Returns:
            [(x, 점수, 최종 동전 리스트), ...] 리스트 (점수 내림차순 정렬)
        """
        results = []
        
        for x in x_positions:
            final_coins, score = self.simulate_drop(
                current_coins,
                drop_coin_type,
                x,
                drop_y
            )
            
            results.append((x, score, final_coins))
        
        # 점수 내림차순 정렬
        results.sort(key=lambda r: r[1], reverse=True)
        
        logger.info(f"다중 시뮬레이션 완료: {len(x_positions)}개 위치 테스트")
        
        return results
    
    def find_best_position(
        self,
        current_coins: List[Coin],
        drop_coin_type: CoinType,
        sample_step: int = 10,
        drop_y: float = 50
    ) -> Tuple[float, float]:
        """
        최적의 낙하 위치 찾기
        
        Args:
            current_coins: 현재 게임 상태
            drop_coin_type: 떨어뜨릴 동전 종류
            sample_step: x좌표 샘플링 간격 (픽셀)
            drop_y: 떨어뜨릴 y 좌표
            
        Returns:
            (최적 x 좌표, 예상 점수)
        """
        # 샘플링할 x 좌표 생성
        radius = drop_coin_type.radius
        min_x = radius + 10
        max_x = self.game_width - radius - 10
        
        x_positions = list(range(int(min_x), int(max_x), sample_step))
        
        if not x_positions:
            logger.warning("샘플링 가능한 x 좌표가 없습니다.")
            return self.game_width / 2, 0.0
        
        # 다중 시뮬레이션
        results = self.simulate_multiple_positions(
            current_coins,
            drop_coin_type,
            x_positions,
            drop_y
        )
        
        # 최고 점수 위치 반환
        best_x, best_score, _ = results[0]
        
        logger.info(f"최적 위치 발견: x={best_x:.1f}, 점수={best_score:.1f}")
        
        return best_x, best_score


# 테스트 코드
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== 물리 시뮬레이터 테스트 ===\n")
    
    # 시뮬레이터 생성
    simulator = PhysicsSimulator(game_width=600, game_height=800)
    print("✅ 시뮬레이터 초기화 완료")
    
    # 테스트 게임 상태 생성
    current_coins = [
        Coin(CoinType.COIN_100, x=200, y=750),
        Coin(CoinType.COIN_100, x=250, y=750),
        Coin(CoinType.COIN_50, x=350, y=750),
    ]
    print(f"\n현재 게임 상태: 동전 {len(current_coins)}개")
    
    # 단일 시뮬레이션 테스트
    print("\n[단일 시뮬레이션 테스트]")
    drop_coin_type = CoinType.COIN_100
    drop_x = 225
    
    final_coins, score = simulator.simulate_drop(
        current_coins,
        drop_coin_type,
        drop_x
    )
    
    print(f"  떨어뜨릴 위치: x={drop_x}")
    print(f"  예상 점수: {score:.1f}")
    print(f"  최종 동전 개수: {len(final_coins)}")
    
    # 최적 위치 찾기 테스트
    print("\n[최적 위치 찾기 테스트]")
    best_x, best_score = simulator.find_best_position(
        current_coins,
        drop_coin_type,
        sample_step=20
    )
    
    print(f"  최적 x 좌표: {best_x:.1f}")
    print(f"  예상 점수: {best_score:.1f}")
    print("\n✅ 모든 테스트 완료")
