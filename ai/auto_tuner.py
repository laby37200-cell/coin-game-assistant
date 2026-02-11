"""
LLM 자동 파라미터 조정 시스템

Gemini API를 사용하여 물리 엔진 파라미터를 자동으로 조정합니다.
예상 결과와 실제 결과를 비교하여 물리 엔진을 최적화합니다.
"""

import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import google.generativeai as genai

from models.coin import Coin, CoinType


logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """시뮬레이션 결과"""
    predicted_coins: List[Coin]  # 예측된 동전 위치
    actual_coins: List[Coin]      # 실제 동전 위치
    drop_x: float                 # 낙하 x 좌표
    drop_coin_type: CoinType      # 떨어뜨린 동전 종류


@dataclass
class PhysicsParameters:
    """물리 엔진 파라미터"""
    gravity: Tuple[float, float]
    friction: float
    elasticity: float
    damping: float
    
    def to_dict(self) -> Dict:
        return {
            'gravity': self.gravity,
            'friction': self.friction,
            'elasticity': self.elasticity,
            'damping': self.damping
        }


class AutoTuner:
    """LLM 기반 자동 파라미터 조정 시스템"""
    
    TUNING_PROMPT_TEMPLATE = """너는 물리 엔진 파라미터 최적화 전문가다.

수박게임(동전게임)의 물리 시뮬레이션 결과와 실제 게임 결과를 비교하여,
물리 엔진 파라미터를 자동으로 조정해야 한다.

## 현재 물리 파라미터
```json
{current_params}
```

## 시뮬레이션 vs 실제 비교

### 예측된 결과 (시뮬레이션)
```json
{predicted_result}
```

### 실제 결과 (게임)
```json
{actual_result}
```

## 차이 분석
{difference_analysis}

## 요구사항

다음 기준으로 물리 파라미터를 조정하라:

1. **중력 (gravity)**: 동전이 떨어지는 속도
   - 실제보다 빠르면 감소 (현재 - 50~100)
   - 실제보다 느리면 증가 (현재 + 50~100)

2. **마찰 (friction)**: 동전이 굴러가는 정도
   - 실제보다 많이 굴러가면 증가 (현재 + 0.05~0.1)
   - 실제보다 적게 굴러가면 감소 (현재 - 0.05~0.1)

3. **탄성 (elasticity)**: 동전이 튕기는 정도
   - 실제보다 많이 튕기면 감소 (현재 - 0.05~0.1)
   - 실제보다 적게 튕기면 증가 (현재 + 0.05~0.1)

4. **감쇠 (damping)**: 에너지 손실 정도
   - 실제보다 빨리 멈추면 증가 (현재 + 0.01~0.02)
   - 실제보다 늦게 멈추면 감소 (현재 - 0.01~0.02)

## 출력 형식 (JSON만)

```json
{
  "adjusted_params": {
    "gravity": [0, new_value],
    "friction": new_value,
    "elasticity": new_value,
    "damping": new_value
  },
  "reasoning": "조정 이유 설명",
  "confidence": 0.0-1.0,
  "expected_improvement": "예상되는 개선 효과"
}
```

**중요**: JSON 형식만 출력하고, 다른 설명은 포함하지 않는다.
"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        """
        Args:
            api_key: Gemini API 키
            model_name: 사용할 모델명
        """
        self.api_key = api_key
        self.model_name = model_name
        
        # Gemini API 초기화
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
        # 파라미터 히스토리
        self.history: List[Tuple[PhysicsParameters, float]] = []
        
        logger.info(f"AutoTuner 초기화: 모델 = {model_name}")
    
    def analyze_difference(
        self,
        predicted_coins: List[Coin],
        actual_coins: List[Coin]
    ) -> str:
        """
        예측과 실제 결과의 차이 분석
        
        Args:
            predicted_coins: 예측된 동전 리스트
            actual_coins: 실제 동전 리스트
            
        Returns:
            차이 분석 텍스트
        """
        analysis = []
        
        # 동전 개수 비교
        analysis.append(f"동전 개수: 예측 {len(predicted_coins)}개, 실제 {len(actual_coins)}개")
        
        # 위치 차이 계산
        if len(predicted_coins) == len(actual_coins):
            total_distance = 0
            for pred, actual in zip(predicted_coins, actual_coins):
                distance = pred.distance_to(actual)
                total_distance += distance
            
            avg_distance = total_distance / len(predicted_coins)
            analysis.append(f"평균 위치 오차: {avg_distance:.2f} 픽셀")
            
            # 오차가 큰 동전 식별
            large_errors = []
            for i, (pred, actual) in enumerate(zip(predicted_coins, actual_coins)):
                distance = pred.distance_to(actual)
                if distance > avg_distance * 1.5:
                    large_errors.append(f"  동전 {i+1}: {distance:.2f}px 오차")
            
            if large_errors:
                analysis.append("큰 오차가 있는 동전:")
                analysis.extend(large_errors)
        
        # 높이 차이 분석
        if predicted_coins and actual_coins:
            pred_max_height = min(c.y for c in predicted_coins)
            actual_max_height = min(c.y for c in actual_coins)
            height_diff = pred_max_height - actual_max_height
            
            analysis.append(f"최고 높이 차이: {height_diff:.2f} 픽셀")
            
            if height_diff > 20:
                analysis.append("→ 시뮬레이션이 실제보다 높게 쌓임 (중력 증가 필요)")
            elif height_diff < -20:
                analysis.append("→ 시뮬레이션이 실제보다 낮게 쌓임 (중력 감소 필요)")
        
        return "\n".join(analysis)
    
    def tune_parameters(
        self,
        current_params: PhysicsParameters,
        simulation_result: SimulationResult
    ) -> Tuple[PhysicsParameters, Dict]:
        """
        LLM을 사용하여 파라미터 자동 조정
        
        Args:
            current_params: 현재 물리 파라미터
            simulation_result: 시뮬레이션 결과
            
        Returns:
            (조정된 파라미터, 조정 정보)
        """
        # 차이 분석
        diff_analysis = self.analyze_difference(
            simulation_result.predicted_coins,
            simulation_result.actual_coins
        )
        
        # 예측/실제 결과를 JSON으로 변환
        predicted_json = json.dumps([c.to_dict() for c in simulation_result.predicted_coins], indent=2, ensure_ascii=False)
        actual_json = json.dumps([c.to_dict() for c in simulation_result.actual_coins], indent=2, ensure_ascii=False)
        
        # 프롬프트 생성
        prompt = self.TUNING_PROMPT_TEMPLATE.format(
            current_params=json.dumps(current_params.to_dict(), indent=2),
            predicted_result=predicted_json,
            actual_result=actual_json,
            difference_analysis=diff_analysis
        )
        
        try:
            # Gemini API 호출
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # JSON 파싱
            # 코드 블록 제거
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(response_text)
            
            # 조정된 파라미터 추출
            adjusted = result['adjusted_params']
            new_params = PhysicsParameters(
                gravity=tuple(adjusted['gravity']),
                friction=adjusted['friction'],
                elasticity=adjusted['elasticity'],
                damping=adjusted['damping']
            )
            
            # 히스토리에 추가
            confidence = result.get('confidence', 0.5)
            self.history.append((new_params, confidence))
            
            logger.info(f"파라미터 자동 조정 완료: confidence={confidence:.2f}")
            logger.info(f"  중력: {current_params.gravity} → {new_params.gravity}")
            logger.info(f"  마찰: {current_params.friction} → {new_params.friction}")
            logger.info(f"  탄성: {current_params.elasticity} → {new_params.elasticity}")
            logger.info(f"  감쇠: {current_params.damping} → {new_params.damping}")
            
            return new_params, result
            
        except Exception as e:
            logger.error(f"파라미터 조정 실패: {e}")
            # 실패 시 현재 파라미터 유지
            return current_params, {'error': str(e)}
    
    def get_best_parameters(self) -> Optional[PhysicsParameters]:
        """
        히스토리에서 가장 신뢰도가 높은 파라미터 반환
        
        Returns:
            최적 파라미터 또는 None
        """
        if not self.history:
            return None
        
        # 신뢰도 기준으로 정렬
        sorted_history = sorted(self.history, key=lambda x: x[1], reverse=True)
        return sorted_history[0][0]
    
    def save_history(self, filepath: str):
        """히스토리를 파일로 저장"""
        history_data = [
            {
                'params': params.to_dict(),
                'confidence': conf
            }
            for params, conf in self.history
        ]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"파라미터 히스토리 저장: {filepath}")
    
    def load_history(self, filepath: str):
        """파일에서 히스토리 로드"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
            
            self.history = [
                (
                    PhysicsParameters(
                        gravity=tuple(item['params']['gravity']),
                        friction=item['params']['friction'],
                        elasticity=item['params']['elasticity'],
                        damping=item['params']['damping']
                    ),
                    item['confidence']
                )
                for item in history_data
            ]
            
            logger.info(f"파라미터 히스토리 로드: {filepath} ({len(self.history)}개)")
            
        except Exception as e:
            logger.error(f"히스토리 로드 실패: {e}")


# 테스트 코드
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== LLM 자동 파라미터 조정 시스템 테스트 ===\n")
    
    # AutoTuner 생성 (API 키 필요)
    import os
    api_key = os.getenv("GEMINI_API_KEY", "")
    
    if not api_key:
        print("❌ GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
        exit(1)
    
    tuner = AutoTuner(api_key)
    print("✅ AutoTuner 초기화 완료\n")
    
    # 테스트 파라미터
    current_params = PhysicsParameters(
        gravity=(0, 900),
        friction=0.5,
        elasticity=0.3,
        damping=0.95
    )
    
    # 테스트 시뮬레이션 결과
    predicted_coins = [
        Coin(CoinType.BLUE_CIRCLE, x=200, y=750),
        Coin(CoinType.YELLOW_CIRCLE, x=250, y=750),
    ]
    
    actual_coins = [
        Coin(CoinType.BLUE_CIRCLE, x=210, y=740),  # 약간 오른쪽, 약간 위
        Coin(CoinType.YELLOW_CIRCLE, x=260, y=745),
    ]
    
    sim_result = SimulationResult(
        predicted_coins=predicted_coins,
        actual_coins=actual_coins,
        drop_x=225,
        drop_coin_type=CoinType.PINK_CIRCLE
    )
    
    print("[차이 분석]")
    diff = tuner.analyze_difference(predicted_coins, actual_coins)
    print(diff)
    print()
    
    print("[파라미터 자동 조정 중...]")
    new_params, info = tuner.tune_parameters(current_params, sim_result)
    
    print("\n[조정 결과]")
    print(f"신뢰도: {info.get('confidence', 0):.2f}")
    print(f"이유: {info.get('reasoning', 'N/A')}")
    print(f"예상 개선: {info.get('expected_improvement', 'N/A')}")
    
    print("\n✅ 테스트 완료")
