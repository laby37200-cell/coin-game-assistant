"""
최적 위치 계산 Optimizer

물리 시뮬레이션과 전략 평가를 결합하여 최적의 동전 낙하 위치를 찾습니다.
"""

import logging
from typing import List, Tuple, Optional
from models.coin import Coin, CoinType
from physics.simulator import PhysicsSimulator
from solver.strategy import StrategyEvaluator
from utils.coordinate_mapper import CoordinateMapper


logger = logging.getLogger(__name__)


class PositionOptimizer:
    """최적 낙하 위치 계산 클래스"""
    
    def __init__(
        self,
        game_width: int,
        game_height: int,
        algorithm: str = "greedy",
        sample_step: int = 10,
        lookahead_depth: int = 1,
        coordinate_mapper: Optional[CoordinateMapper] = None,
        physics_params: Optional[dict] = None
    ):
        """
        Args:
            game_width: 게임 영역 너비
            game_height: 게임 영역 높이
            algorithm: 최적화 알고리즘 ("greedy" 또는 "monte_carlo")
            sample_step: x좌표 샘플링 간격 (픽셀)
            lookahead_depth: Look-ahead 깊이 (1=현재만, 2=다음 턴까지)
        """
        self.game_width = game_width
        self.game_height = game_height
        self.algorithm = algorithm
        self.sample_step = sample_step
        self.lookahead_depth = lookahead_depth
        self.coordinate_mapper = coordinate_mapper
        self.physics_params = physics_params or {}
        
        # 물리 시뮬레이터
        self.simulator = PhysicsSimulator(
            game_width,
            game_height,
            coordinate_mapper=coordinate_mapper,
            **self.physics_params,
        )
        
        # 전략 평가자
        self.evaluator = StrategyEvaluator(game_width, game_height)
        
        logger.info(f"PositionOptimizer 초기화: 알고리즘={algorithm}, "
                   f"샘플링={sample_step}px, lookahead={lookahead_depth}")
    
    def find_optimal_position(
        self,
        current_coins: List[Coin],
        drop_coin_type: CoinType,
        next_coin_type: Optional[CoinType] = None
    ) -> Tuple[float, float, dict]:
        """
        최적의 낙하 위치 찾기
        
        Args:
            current_coins: 현재 게임 상태의 동전 리스트
            drop_coin_type: 떨어뜨릴 동전 종류
            next_coin_type: 다음에 올 동전 종류 (lookahead용)
            
        Returns:
            (최적 x 좌표, 예상 점수, 상세 정보)
        """
        if self.algorithm == "greedy":
            return self._greedy_search(current_coins, drop_coin_type, next_coin_type)
        elif self.algorithm == "monte_carlo":
            return self._monte_carlo_search(current_coins, drop_coin_type, next_coin_type)
        else:
            logger.warning(f"알 수 없는 알고리즘: {self.algorithm}, greedy 사용")
            return self._greedy_search(current_coins, drop_coin_type, next_coin_type)
    
    def _greedy_search(
        self,
        current_coins: List[Coin],
        drop_coin_type: CoinType,
        next_coin_type: Optional[CoinType] = None
    ) -> Tuple[float, float, dict]:
        """
        Greedy Search: 모든 가능한 위치를 시뮬레이션하여 최고 점수 선택
        
        Args:
            current_coins: 현재 게임 상태
            drop_coin_type: 떨어뜨릴 동전
            next_coin_type: 다음 동전 (사용 안 함)
            
        Returns:
            (최적 x, 점수, 상세 정보)
        """
        # 샘플링할 x 좌표 생성
        x_positions = self._generate_sample_positions(drop_coin_type)
        
        if not x_positions:
            logger.warning("샘플링 가능한 위치가 없습니다.")
            return self.game_width / 2, 0.0, {}
        
        best_x = x_positions[0]
        best_score = float('-inf')
        best_coins = []
        
        # 모든 위치 시뮬레이션
        for x in x_positions:
            # 물리 시뮬레이션
            final_coins, sim_score = self.simulator.simulate_drop(
                current_coins,
                drop_coin_type,
                x
            )
            
            # 전략 평가
            strategy_score = self.evaluator.evaluate(final_coins)
            score = sim_score + strategy_score
            
            # 최고 점수 업데이트
            if score > best_score:
                best_score = score
                best_x = x
                best_coins = final_coins
        
        # 상세 정보
        details = {
            'positions_tested': len(x_positions),
            'best_position': best_x,
            'best_score': best_score,
            'evaluation_details': self.evaluator.get_evaluation_details(best_coins)
        }
        
        logger.info(f"Greedy Search 완료: 최적 x={best_x:.1f}, 점수={best_score:.1f}")
        
        return best_x, best_score, details
    
    def _monte_carlo_search(
        self,
        current_coins: List[Coin],
        drop_coin_type: CoinType,
        next_coin_type: Optional[CoinType] = None
    ) -> Tuple[float, float, dict]:
        """
        Monte Carlo Search: 랜덤 샘플링 + 평균 점수 기반
        (현재는 Greedy와 동일하게 구현, 필요시 확장 가능)
        
        Args:
            current_coins: 현재 게임 상태
            drop_coin_type: 떨어뜨릴 동전
            next_coin_type: 다음 동전
            
        Returns:
            (최적 x, 점수, 상세 정보)
        """
        # 현재는 Greedy와 동일
        return self._greedy_search(current_coins, drop_coin_type, next_coin_type)
    
    def _generate_sample_positions(self, drop_coin_type: CoinType) -> List[float]:
        """
        샘플링할 x 좌표 리스트 생성
        
        Args:
            drop_coin_type: 떨어뜨릴 동전 종류
            
        Returns:
            x 좌표 리스트
        """
        radius = drop_coin_type.radius
        
        # 벽에서 동전 반지름만큼 떨어진 위치부터 시작
        min_x = radius + 10
        max_x = self.game_width - radius - 10
        
        # 샘플링 간격으로 위치 생성
        positions = []
        x = min_x
        
        while x <= max_x:
            positions.append(x)
            x += self.sample_step
        
        # 마지막 위치 추가
        if positions[-1] < max_x:
            positions.append(max_x)
        
        logger.debug(f"샘플링 위치 생성: {len(positions)}개 ({min_x:.1f} ~ {max_x:.1f})")
        
        return positions
    
    def evaluate_position(
        self,
        current_coins: List[Coin],
        drop_coin_type: CoinType,
        drop_x: float
    ) -> Tuple[float, List[Coin], dict]:
        """
        특정 위치의 평가 점수 계산
        
        Args:
            current_coins: 현재 게임 상태
            drop_coin_type: 떨어뜨릴 동전
            drop_x: 평가할 x 좌표
            
        Returns:
            (점수, 시뮬레이션 후 동전 리스트, 평가 상세)
        """
        # 물리 시뮬레이션
        final_coins, sim_score = self.simulator.simulate_drop(
            current_coins,
            drop_coin_type,
            drop_x
        )
        
        # 전략 평가
        strategy_score = self.evaluator.evaluate(final_coins)
        score = sim_score + strategy_score
        details = self.evaluator.get_evaluation_details(final_coins)
        
        return score, final_coins, details
    
    def compare_positions(
        self,
        current_coins: List[Coin],
        drop_coin_type: CoinType,
        positions: List[float]
    ) -> List[Tuple[float, float, dict]]:
        """
        여러 위치를 비교하여 점수 순으로 정렬
        
        Args:
            current_coins: 현재 게임 상태
            drop_coin_type: 떨어뜨릴 동전
            positions: 비교할 x 좌표 리스트
            
        Returns:
            [(x, 점수, 상세), ...] 리스트 (점수 내림차순)
        """
        results = []
        
        for x in positions:
            score, _, details = self.evaluate_position(
                current_coins,
                drop_coin_type,
                x
            )
            results.append((x, score, details))
        
        # 점수 내림차순 정렬
        results.sort(key=lambda r: r[1], reverse=True)
        
        return results


# 테스트 코드
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== 최적 위치 계산 Optimizer 테스트 ===\n")
    
    # Optimizer 생성
    optimizer = PositionOptimizer(
        game_width=600,
        game_height=800,
        algorithm="greedy",
        sample_step=20
    )
    print("✅ Optimizer 초기화 완료")
    
    # 테스트 게임 상태
    current_coins = [
        Coin(CoinType.YELLOW_CIRCLE, x=200, y=750),
        Coin(CoinType.YELLOW_CIRCLE, x=250, y=750),
        Coin(CoinType.ORANGE_CIRCLE, x=350, y=750),
    ]
    print(f"\n현재 게임 상태: 동전 {len(current_coins)}개")
    
    # 떨어뜨릴 동전
    drop_coin = CoinType.YELLOW_CIRCLE
    print(f"떨어뜨릴 동전: {drop_coin.display_name}")
    
    # 최적 위치 찾기
    print("\n최적 위치 계산 중...")
    best_x, best_score, details = optimizer.find_optimal_position(
        current_coins,
        drop_coin
    )
    
    print(f"\n✅ 최적 위치 발견!")
    print(f"  x 좌표: {best_x:.1f}")
    print(f"  예상 점수: {best_score:.1f}")
    print(f"  테스트한 위치: {details['positions_tested']}개")
    
    print("\n평가 상세:")
    for key, value in details['evaluation_details'].items():
        if key != 'total':
            print(f"  {key}: {value:.1f}")
    
    # 특정 위치 비교
    print("\n[위치 비교 테스트]")
    test_positions = [200, 225, 250, 300]
    comparison = optimizer.compare_positions(current_coins, drop_coin, test_positions)
    
    print("위치별 점수:")
    for x, score, _ in comparison:
        print(f"  x={x:.0f}: {score:.1f}")
    
    print("\n✅ 모든 테스트 완료")
