"""
Gemini Vision API 통합 모듈

Google Gemini 3.0 Flash Preview 모델을 사용하여 게임 화면을 분석하고
동전의 위치, 종류, 크기를 추출합니다.
"""

import json
import base64
import io
import logging
from typing import List, Dict, Optional
from PIL import Image
import google.generativeai as genai

from models.coin import Coin, CoinType


logger = logging.getLogger(__name__)


class GeminiAnalyzer:
    """Gemini API를 사용한 화면 분석 클래스"""
    
    # Gemini에게 전달할 시스템 프롬프트
    SYSTEM_PROMPT = """너는 '콘텐츠페이 동전게임'(수박게임의 변형판) 전문 분석가다.

게임 화면 이미지를 분석하여 다음 정보를 JSON 형식으로 정확하게 반환해야 한다:

1. **현재 화면에 있는 모든 동전의 정보**:
   - 동전 종류: 검정번개, 핑크동전, 주황동전, 노랑동전, 민트동전, 파랑동전, 보라동전, 갈색동전, 흰색상자, 노랑전구, 민트선물상자
   - 중심 좌표 (x, y) - 픽셀 단위
   - 반지름 (radius) - 픽셀 단위

2. **다음에 떨어질 동전 (Next)**: 
   - 화면 상단에 표시된 "다음 동전"의 종류
   - 랜덤으로 떨어지는 동전은 검정번개~파랑동전까지만 가능

3. **게임 영역 크기**:
   - 너비 (width)
   - 높이 (height)

**동전 크기 참고**:
- 검정번개 (가장 작음, 약 20px 반지름)
- 핑크동전 (약 25px)
- 주황동전 (약 30px)
- 노랑동전 (약 35px)
- 민트동전 (약 40px)
- 파랑동전 (약 50px)
- 보라동전 (약 60px)
- 갈색동전 (약 70px)
- 흰색상자 (약 85px)
- 노랑전구 (약 100px)
- 민트선물상자 (가장 큼, 약 120px 반지름)

**출력 형식 (JSON)**:
```json
{
  "coins": [
    {
      "type": "동전종류",
      "x": 중심x좌표,
      "y": 중심y좌표,
      "radius": 반지름
    },
    ...
  ],
  "next_coin": "다음동전종류",
  "game_area": {
    "width": 너비,
    "height": 높이
  }
}
```

**중요 규칙**:
- 좌표는 이미지의 왼쪽 상단을 (0, 0)으로 하는 픽셀 좌표계를 사용한다.
- 동전의 크기는 종류에 따라 다르다: 검정번개(작음) → 민트선물상자(큼)
- 모든 동전을 빠짐없이 정확하게 감지해야 한다.
- 겹쳐진 동전도 최대한 정확하게 분리하여 인식한다.
- JSON 형식만 출력하고, 다른 설명은 포함하지 않는다.
- 동전 색상 참고: 검정(번개), 핑크, 주황, 노랑, 민트, 파랑, 보라, 갈색, 흰색, 노랑(전구), 민트(선물상자)
"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        """
        Args:
            api_key: Google Gemini API 키
            model_name: 사용할 모델명
        """
        self.api_key = api_key
        self.model_name = model_name
        
        # Gemini API 초기화
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
        logger.info(f"GeminiAnalyzer 초기화: 모델 = {model_name}")
    
    def analyze_image(self, image: Image.Image) -> Optional[Dict]:
        """
        이미지 분석하여 동전 정보 추출
        
        Args:
            image: PIL Image 객체
            
        Returns:
            분석 결과 딕셔너리 또는 None
            {
                'coins': [{'type': str, 'x': float, 'y': float, 'radius': float}, ...],
                'next_coin': str,
                'game_area': {'width': int, 'height': int}
            }
        """
        try:
            logger.info("Gemini API로 이미지 분석 시작...")
            
            # 프롬프트 구성
            prompt = self.SYSTEM_PROMPT + "\n\n이미지를 분석하고 JSON 형식으로 결과를 반환하라."
            
            # Gemini API 호출
            response = self.model.generate_content([prompt, image])
            
            # 응답 텍스트 추출
            response_text = response.text.strip()
            logger.debug(f"Gemini 응답: {response_text}")
            
            # JSON 파싱
            result = self._parse_json_response(response_text)
            
            if result:
                logger.info(f"분석 완료: 동전 {len(result.get('coins', []))}개 감지")
                return result
            else:
                logger.error("JSON 파싱 실패")
                return None
                
        except Exception as e:
            logger.error(f"이미지 분석 실패: {e}")
            return None
    
    def _parse_json_response(self, response_text: str) -> Optional[Dict]:
        """
        Gemini 응답에서 JSON 추출 및 파싱
        
        Args:
            response_text: Gemini 응답 텍스트
            
        Returns:
            파싱된 딕셔너리 또는 None
        """
        try:
            # JSON 코드 블록 제거 (```json ... ``` 형식)
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            
            # JSON 파싱
            data = json.loads(response_text)
            
            # 필수 필드 검증
            if 'coins' not in data:
                logger.warning("응답에 'coins' 필드가 없습니다.")
                data['coins'] = []
            
            if 'next_coin' not in data:
                logger.warning("응답에 'next_coin' 필드가 없습니다.")
                data['next_coin'] = None
            
            if 'game_area' not in data:
                logger.warning("응답에 'game_area' 필드가 없습니다.")
                data['game_area'] = {'width': 600, 'height': 800}
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 오류: {e}")
            logger.debug(f"응답 텍스트: {response_text}")
            return None
        except Exception as e:
            logger.error(f"응답 처리 오류: {e}")
            return None
    
    def extract_coins(self, analysis_result: Dict) -> List[Coin]:
        """
        분석 결과에서 Coin 객체 리스트 생성
        
        Args:
            analysis_result: analyze_image()의 반환값
            
        Returns:
            Coin 객체 리스트
        """
        coins = []
        
        for coin_data in analysis_result.get('coins', []):
            try:
                # 동전 타입 찾기
                coin_type = CoinType.from_name(coin_data['type'])
                
                if not coin_type:
                    logger.warning(f"알 수 없는 동전 타입: {coin_data['type']}")
                    continue
                
                # Coin 객체 생성
                coin = Coin(
                    coin_type=coin_type,
                    x=float(coin_data['x']),
                    y=float(coin_data['y'])
                )
                
                coins.append(coin)
                
            except (KeyError, ValueError) as e:
                logger.warning(f"동전 데이터 파싱 오류: {e}, 데이터: {coin_data}")
                continue
        
        return coins
    
    def get_next_coin_type(self, analysis_result: Dict) -> Optional[CoinType]:
        """
        분석 결과에서 다음 동전 타입 추출
        
        Args:
            analysis_result: analyze_image()의 반환값
            
        Returns:
            다음 동전의 CoinType 또는 None
        """
        next_coin_name = analysis_result.get('next_coin')
        
        if not next_coin_name:
            return None
        
        return CoinType.from_name(next_coin_name)
    
    def analyze_and_extract(self, image: Image.Image) -> tuple[List[Coin], Optional[CoinType], Dict]:
        """
        이미지 분석 및 동전 추출을 한 번에 수행
        
        Args:
            image: PIL Image 객체
            
        Returns:
            (동전 리스트, 다음 동전 타입, 게임 영역 정보)
        """
        # 이미지 분석
        result = self.analyze_image(image)
        
        if not result:
            return [], None, {}
        
        # 동전 추출
        coins = self.extract_coins(result)
        
        # 다음 동전 타입
        next_coin = self.get_next_coin_type(result)
        
        # 게임 영역
        game_area = result.get('game_area', {})
        
        return coins, next_coin, game_area


# 테스트 코드
if __name__ == "__main__":
    import os
    
    logging.basicConfig(level=logging.INFO)
    
    print("=== Gemini Vision API 테스트 ===\n")
    
    # API 키 확인
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("   export GEMINI_API_KEY='your-api-key' 를 실행하세요.")
        exit(1)
    
    # GeminiAnalyzer 인스턴스 생성
    analyzer = GeminiAnalyzer(api_key)
    
    # 테스트 이미지 생성 (더미)
    print("테스트 이미지 생성 중...")
    test_image = Image.new('RGB', (600, 800), color=(240, 240, 240))
    
    # 분석 테스트
    print("\nGemini API 호출 중...")
    result = analyzer.analyze_image(test_image)
    
    if result:
        print("✅ 분석 성공")
        print(f"\n결과:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # 동전 추출
        coins = analyzer.extract_coins(result)
        print(f"\n추출된 동전: {len(coins)}개")
        for coin in coins:
            print(f"  - {coin.coin_type.display_name} at ({coin.x:.1f}, {coin.y:.1f})")
        
        # 다음 동전
        next_coin = analyzer.get_next_coin_type(result)
        if next_coin:
            print(f"\n다음 동전: {next_coin.display_name}")
    else:
        print("❌ 분석 실패")
