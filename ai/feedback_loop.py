"""
물리 엔진 자동 튜닝 및 피드백 루프 시스템

실제 게임 결과와 시뮬레이션 결과를 비교하여
물리 엔진 파라미터를 자동으로 최적화합니다.
"""

import logging
import time
from typing import List, Dict, Optional
from dataclasses import dataclass

from models.coin import Coin, CoinType
from ai.auto_tuner import AutoTuner, PhysicsParameters, SimulationResult
from physics.simulator import PhysicsSimulator


logger = logging.getLogger(__name__)


@dataclass
class FeedbackData:
    """피드백 데이터"""
    timestamp: float
    before_state: List[Coin]
    drop_x: float
    drop_coin: CoinType
    predicted_state: List[Coin]
    actual_state: List[Coin]
    accuracy: float  # 0.0 ~ 1.0


class FeedbackLoop:
    """물리 엔진 자동 튜닝 피드백 루프"""
    
    def __init__(
        self,
        auto_tuner: AutoTuner,
        simulator: PhysicsSimulator,
        min_accuracy: float = 0.8,
        max_iterations: int = 10
    ):
        """
        Args:
            auto_tuner: 자동 튜너
            simulator: 물리 시뮬레이터
            min_accuracy: 최소 정확도 (이 값 이상이면 튜닝 중단)
            max_iterations: 최대 반복 횟수
        """
        self.auto_tuner = auto_tuner
        self.simulator = simulator
        self.min_accuracy = min_accuracy
        self.max_iterations = max_iterations
        
        # 피드백 히스토리
        self.feedback_history: List[FeedbackData] = []
        
        logger.info(f"FeedbackLoop 초기화: min_accuracy={min_accuracy}, max_iterations={max_iterations}")
    
    def calculate_accuracy(
        self,
        predicted_coins: List[Coin],
        actual_coins: List[Coin]
    ) -> float:
        """
        예측 정확도 계산
        
        Args:
            predicted_coins: 예측된 동전 리스트
            actual_coins: 실제 동전 리스트
            
        Returns:
            정확도 (0.0 ~ 1.0)
        """
        if not predicted_coins or not actual_coins:
            return 0.0
        
        if len(predicted_coins) != len(actual_coins):
            # 개수가 다르면 페널티
            count_penalty = abs(len(predicted_coins) - len(actual_coins)) * 0.1
            return max(0.0, 1.0 - count_penalty)
        
        # 각 동전의 위치 오차 계산
        total_error = 0.0
        for pred, actual in zip(predicted_coins, actual_coins):
            distance = pred.distance_to(actual)
            # 거리를 정확도로 변환 (가까울수록 1.0에 가까움)
            # 50px 이상 차이나면 0.0
            accuracy = max(0.0, 1.0 - (distance / 50.0))
            total_error += accuracy
        
        # 평균 정확도
        avg_accuracy = total_error / len(predicted_coins)
        return avg_accuracy
    
    def record_feedback(
        self,
        before_state: List[Coin],
        drop_x: float,
        drop_coin: CoinType,
        predicted_state: List[Coin],
        actual_state: List[Coin]
    ):
        """
        피드백 데이터 기록
        
        Args:
            before_state: 동전 떨어뜨리기 전 상태
            drop_x: 낙하 x 좌표
            drop_coin: 떨어뜨린 동전 종류
            predicted_state: 예측된 상태
            actual_state: 실제 상태
        """
        accuracy = self.calculate_accuracy(predicted_state, actual_state)
        
        feedback = FeedbackData(
            timestamp=time.time(),
            before_state=before_state,
            drop_x=drop_x,
            drop_coin=drop_coin,
            predicted_state=predicted_state,
            actual_state=actual_state,
            accuracy=accuracy
        )
        
        self.feedback_history.append(feedback)
        
        logger.info(f"피드백 기록: 정확도={accuracy:.2f}")
    
    def should_tune(self) -> bool:
        """
        튜닝이 필요한지 판단
        
        Returns:
            튜닝 필요 여부
        """
        if not self.feedback_history:
            return False
        
        # 최근 5개의 평균 정확도 확인
        recent_feedbacks = self.feedback_history[-5:]
        avg_accuracy = sum(f.accuracy for f in recent_feedbacks) / len(recent_feedbacks)
        
        if avg_accuracy < self.min_accuracy:
            logger.info(f"튜닝 필요: 평균 정확도 {avg_accuracy:.2f} < {self.min_accuracy}")
            return True
        
        return False
    
    def auto_tune_loop(self) -> PhysicsParameters:
        """
        자동 튜닝 루프 실행
        
        Returns:
            최적화된 물리 파라미터
        """
        if not self.feedback_history:
            logger.warning("피드백 데이터가 없습니다.")
            return self.simulator.get_current_parameters()
        
        current_params = self.simulator.get_current_parameters()
        best_params = current_params
        best_accuracy = 0.0
        
        for iteration in range(self.max_iterations):
            logger.info(f"\n=== 튜닝 반복 {iteration + 1}/{self.max_iterations} ===")
            
            # 최근 피드백 데이터 사용
            recent_feedback = self.feedback_history[-1]
            
            # 시뮬레이션 결과 생성
            sim_result = SimulationResult(
                predicted_coins=recent_feedback.predicted_state,
                actual_coins=recent_feedback.actual_state,
                drop_x=recent_feedback.drop_x,
                drop_coin_type=recent_feedback.drop_coin
            )
            
            # 파라미터 조정
            new_params, info = self.auto_tuner.tune_parameters(current_params, sim_result)
            
            # 새 파라미터로 시뮬레이터 업데이트
            self.simulator.update_parameters(new_params)
            
            # 재시뮬레이션
            new_predicted, _ = self.simulator.simulate_drop(
                recent_feedback.before_state,
                recent_feedback.drop_coin,
                recent_feedback.drop_x
            )
            
            # 정확도 계산
            new_accuracy = self.calculate_accuracy(new_predicted, recent_feedback.actual_state)
            
            logger.info(f"정확도: {recent_feedback.accuracy:.2f} → {new_accuracy:.2f}")
            
            # 개선되었으면 업데이트
            if new_accuracy > best_accuracy:
                best_params = new_params
                best_accuracy = new_accuracy
                logger.info(f"✅ 개선됨! 최고 정확도: {best_accuracy:.2f}")
            
            # 목표 정확도 달성 시 종료
            if new_accuracy >= self.min_accuracy:
                logger.info(f"🎯 목표 정확도 달성: {new_accuracy:.2f} >= {self.min_accuracy}")
                break
            
            current_params = new_params
        
        # 최적 파라미터로 시뮬레이터 업데이트
        self.simulator.update_parameters(best_params)
        logger.info(f"\n최종 정확도: {best_accuracy:.2f}")
        
        return best_params
    
    def get_average_accuracy(self, last_n: int = 10) -> float:
        """
        최근 N개의 평균 정확도 반환
        
        Args:
            last_n: 최근 N개
            
        Returns:
            평균 정확도
        """
        if not self.feedback_history:
            return 0.0
        
        recent = self.feedback_history[-last_n:]
        return sum(f.accuracy for f in recent) / len(recent)
    
    def get_statistics(self) -> Dict:
        """
        통계 정보 반환
        
        Returns:
            통계 딕셔너리
        """
        if not self.feedback_history:
            return {
                'total_feedbacks': 0,
                'average_accuracy': 0.0,
                'best_accuracy': 0.0,
                'worst_accuracy': 0.0
            }
        
        accuracies = [f.accuracy for f in self.feedback_history]
        
        return {
            'total_feedbacks': len(self.feedback_history),
            'average_accuracy': sum(accuracies) / len(accuracies),
            'best_accuracy': max(accuracies),
            'worst_accuracy': min(accuracies),
            'recent_10_avg': self.get_average_accuracy(10)
        }


# 테스트 코드
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== 물리 엔진 피드백 루프 테스트 ===\n")
    
    # AutoTuner와 Simulator 생성 (실제 사용 시 필요)
    import os
    from config import GEMINI_API_KEY
    
    api_key = os.getenv("GEMINI_API_KEY", GEMINI_API_KEY)
    
    if not api_key:
        print("❌ GEMINI_API_KEY가 설정되지 않았습니다.")
        exit(1)
    
    tuner = AutoTuner(api_key)
    simulator = PhysicsSimulator()
    
    # FeedbackLoop 생성
    feedback_loop = FeedbackLoop(tuner, simulator, min_accuracy=0.85)
    print("✅ FeedbackLoop 초기화 완료\n")
    
    # 테스트 피드백 데이터
    before_state = [
        Coin(CoinType.BLUE_CIRCLE, x=200, y=750),
        Coin(CoinType.YELLOW_CIRCLE, x=300, y=750),
    ]
    
    predicted_state = [
        Coin(CoinType.BLUE_CIRCLE, x=200, y=750),
        Coin(CoinType.YELLOW_CIRCLE, x=300, y=750),
        Coin(CoinType.PINK_CIRCLE, x=250, y=720),
    ]
    
    actual_state = [
        Coin(CoinType.BLUE_CIRCLE, x=205, y=745),
        Coin(CoinType.YELLOW_CIRCLE, x=305, y=748),
        Coin(CoinType.PINK_CIRCLE, x=255, y=715),
    ]
    
    # 피드백 기록
    feedback_loop.record_feedback(
        before_state=before_state,
        drop_x=250,
        drop_coin=CoinType.PINK_CIRCLE,
        predicted_state=predicted_state,
        actual_state=actual_state
    )
    
    # 통계 출력
    stats = feedback_loop.get_statistics()
    print("\n[통계]")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ 테스트 완료")
