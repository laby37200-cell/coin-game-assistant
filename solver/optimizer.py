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
        next_coin_type: Optional[CoinType] = None,
        wall_left_x: Optional[float] = None,
        wall_right_x: Optional[float] = None
    ) -> Tuple[float, float, dict]:
        """
        최적의 낙하 위치 찾기
        
        Args:
            current_coins: 현재 게임 상태의 동전 리스트
            drop_coin_type: 떨어뜨릴 동전 종류
            next_coin_type: 다음에 올 동전 종류 (lookahead용)
            wall_left_x: 왼쪽 벽 안쪽 x좌표 (드롭 가능 영역 왼쪽 경계)
            wall_right_x: 오른쪽 벽 안쪽 x좌표 (드롭 가능 영역 오른쪽 경계)
            
        Returns:
            (최적 x 좌표, 예상 점수, 상세 정보)
        """
        if self.algorithm == "greedy":
            return self._greedy_search(current_coins, drop_coin_type, next_coin_type, wall_left_x, wall_right_x)
        elif self.algorithm == "monte_carlo":
            return self._monte_carlo_search(current_coins, drop_coin_type, next_coin_type, wall_left_x, wall_right_x)
        else:
            logger.warning(f"알 수 없는 알고리즘: {self.algorithm}, greedy 사용")
            return self._greedy_search(current_coins, drop_coin_type, next_coin_type, wall_left_x, wall_right_x)
    
    # 2단계 lookahead에서 상위 후보 수
    TOP_K_CANDIDATES = 5
    # 2단계에서 다음 동전 샘플링 간격 (더 넓게)
    LOOKAHEAD_SAMPLE_STEP = 30

    def _greedy_search(
        self,
        current_coins: List[Coin],
        drop_coin_type: CoinType,
        next_coin_type: Optional[CoinType] = None,
        wall_left_x: Optional[float] = None,
        wall_right_x: Optional[float] = None
    ) -> Tuple[float, float, dict]:
        """
        2-Phase Deep Search (오목 AI식 탐색):
          Phase 1: 모든 위치를 시뮬레이션하여 상위 K개 후보 선정
          Phase 2: 각 후보에 대해 다음 동전까지 시뮬레이션 (lookahead)
        
        Args:
            current_coins: 현재 게임 상태
            drop_coin_type: 떨어뜨릴 동전
            next_coin_type: 다음 동전 (lookahead용)
            
        Returns:
            (최적 x, 점수, 상세 정보)
        """
        # Phase 1: 전체 위치 스캔
        x_positions = self._generate_sample_positions(drop_coin_type, wall_left_x, wall_right_x)
        
        if not x_positions:
            logger.warning("샘플링 가능한 위치가 없습니다.")
            center = ((wall_left_x or 0) + (wall_right_x or self.game_width)) / 2
            return center, 0.0, {}
        
        # Phase 1 결과 수집
        candidates = []
        for x in x_positions:
            final_coins, sim_score = self.simulator.simulate_drop(
                current_coins, drop_coin_type, x
            )
            strategy_score = self.evaluator.evaluate(final_coins)
            total = sim_score + strategy_score
            candidates.append((x, total, final_coins, sim_score))
        
        # 상위 K개 후보 선정
        candidates.sort(key=lambda c: c[1], reverse=True)
        top_k = candidates[:self.TOP_K_CANDIDATES]
        
        # Phase 2: Lookahead (다음 동전까지 시뮬레이션)
        best_x = top_k[0][0]
        best_score = top_k[0][1]
        best_coins = top_k[0][2]
        
        if next_coin_type and self.lookahead_depth >= 2:
            lookahead_positions = self._generate_sample_positions(
                next_coin_type, wall_left_x, wall_right_x
            )
            # lookahead는 더 넓은 간격으로 샘플링
            lookahead_positions = lookahead_positions[::max(1, self.LOOKAHEAD_SAMPLE_STEP // max(1, self.sample_step))]
            
            for x1, score1, coins_after_drop1, sim1 in top_k:
                # 각 후보에 대해 다음 동전의 최선 위치 탐색
                best_next_score = float('-inf')
                
                for x2 in lookahead_positions:
                    final2, sim2 = self.simulator.simulate_drop(
                        coins_after_drop1, next_coin_type, x2
                    )
                    strat2 = self.evaluator.evaluate(final2)
                    next_score = sim2 + strat2
                    if next_score > best_next_score:
                        best_next_score = next_score
                
                # 현재 수 + 다음 수의 최선 결과를 합산
                combined = score1 * 0.6 + best_next_score * 0.4
                
                if combined > best_score:
                    best_score = combined
                    best_x = x1
                    best_coins = coins_after_drop1
            
            logger.info(f"Deep Search (2-step): 최적 x={best_x:.1f}, "
                       f"combined={best_score:.1f}, "
                       f"candidates={len(top_k)}, lookahead_pos={len(lookahead_positions)}")
        else:
            logger.info(f"Greedy Search: 최적 x={best_x:.1f}, 점수={best_score:.1f}")
        
        details = {
            'positions_tested': len(x_positions),
            'lookahead_depth': 2 if (next_coin_type and self.lookahead_depth >= 2) else 1,
            'top_k_candidates': len(top_k),
            'best_position': best_x,
            'best_score': best_score,
            'evaluation_details': self.evaluator.get_evaluation_details(best_coins)
        }
        
        return best_x, best_score, details
    
    def _monte_carlo_search(
        self,
        current_coins: List[Coin],
        drop_coin_type: CoinType,
        next_coin_type: Optional[CoinType] = None,
        wall_left_x: Optional[float] = None,
        wall_right_x: Optional[float] = None
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
        return self._greedy_search(current_coins, drop_coin_type, next_coin_type, wall_left_x, wall_right_x)
    
    def _generate_sample_positions(
        self,
        drop_coin_type: CoinType,
        wall_left_x: Optional[float] = None,
        wall_right_x: Optional[float] = None
    ) -> List[float]:
        """
        샘플링할 x 좌표 리스트 생성
        
        Args:
            drop_coin_type: 떨어뜨릴 동전 종류
            wall_left_x: 왼쪽 벽 안쪽 x좌표
            wall_right_x: 오른쪽 벽 안쪽 x좌표
            
        Returns:
            x 좌표 리스트
        """
        radius = drop_coin_type.radius
        
        # 벽 경계 사용 (없으면 게임 전체 너비 기준)
        left_bound = wall_left_x if wall_left_x is not None else 0
        right_bound = wall_right_x if wall_right_x is not None else self.game_width
        
        # 벽에서 동전 반지름만큼 떨어진 위치부터 시작
        min_x = left_bound + radius + 5
        max_x = right_bound - radius - 5
        
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
