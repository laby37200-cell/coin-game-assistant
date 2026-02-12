"""
Gemini Vision API 통합 모듈

Google Gemini 3.0 Flash Preview 모델을 사용하여 게임 화면을 분석하고
동전의 위치, 종류, 크기를 추출합니다.
"""

import json
import logging
import time
from collections import deque
from typing import List, Dict, Optional
from PIL import Image

try:
    # New Google Gen AI SDK
    from google import genai  # type: ignore
    _HAS_NEW_SDK = True
except Exception:
    genai = None
    _HAS_NEW_SDK = False

try:
    # Legacy SDK (google-generativeai)
    import google.generativeai as legacy_genai  # type: ignore
    _HAS_LEGACY_SDK = True
except Exception:
    legacy_genai = None
    _HAS_LEGACY_SDK = False

from models.coin import Coin, CoinType


logger = logging.getLogger(__name__)


class GeminiAnalyzer:
    """Gemini API를 사용한 화면 분석 클래스"""
    
    # Gemini에게 전달할 시스템 프롬프트
    SYSTEM_PROMPT = """너는 dailygame.kr 의 '콘텐츠페이 동전게임'(수박게임 변형) 화면 분석 전문가다.

**게임 화면 구조** (MuMu 에뮬레이터 캡처 이미지 기준):
- 이미지 최상단: 안드로이드 상태바 + 브라우저 주소창 (분석 대상 아님)
- 그 아래: 게임 UI 상단 바 (HOME 버튼, 점수, 코인 수)
- **현재 동전**: 게임 상단 중앙에 크게 표시된 동전 — 지금 떨어뜨릴 동전
- **다음 동전**: 현재 동전 오른쪽에 작게 표시된 동전 — 다음 차례 동전
- **양쪽 벽**: 갈색 벽돌 무늬 세로 벽이 좌우에 있음. 벽 안쪽만 드롭 가능
- **바닥**: 벽 하단에 초록색 바닥
- **게임 플레이 영역**: 양쪽 벽 사이의 하늘색 공간 (동전이 쌓이는 곳)

**분석 항목**:
1. **현재 떨어뜨릴 동전 (current_coin)**: 화면 상단 중앙의 큰 동전 종류
2. **다음 동전 (next_coin)**: 현재 동전 오른쪽의 작은 동전 종류
3. **바닥에 쌓인 동전들 (coins)**: 벽 사이 게임 영역 안에 있는 모든 동전
   - 동전이 하나도 없으면 빈 배열 []
4. **벽 위치 (wall_left_x, wall_right_x)**: 왼쪽 벽 안쪽 경계 x좌표, 오른쪽 벽 안쪽 경계 x좌표

**동전 종류** (작은 것부터 큰 것 순서):
- 검정번개 (가장 작음, ~20px 반지름, 검정색 번개 모양)
- 핑크동전 (~25px, 분홍색 하트)
- 주황동전 (~30px, 주황색)
- 노랑동전 (~35px, 노란색 별)
- 민트동전 (~40px, 초록색 클로버)
- 파랑동전 (~50px, 파란색 다이아몬드)
- 보라동전 (~60px, 보라색)
- 갈색동전 (~70px, 갈색)
- 흰색상자 (~85px, 흰색 상자)
- 노랑전구 (~100px, 노란색 전구)
- 민트선물상자 (가장 큼, ~120px, 민트색 선물상자)

5. **실제 게임 점수 (game_score)**: 상단 바의 노란 동전 아이콘 옆에 표시된 숫자 (예: 499)
6. **천장 라인 y좌표 (ceiling_y)**: 흰색 점선의 y좌표. 이 위로 동전이 올라가면 게임오버.

**출력 형식 (JSON만 출력)**:
```json
{
  "current_coin": "현재동전종류",
  "next_coin": "다음동전종류",
  "coins": [
    {"type": "동전종류", "x": 중심x, "y": 중심y, "radius": 반지름},
    ...
  ],
  "wall_left_x": 왼쪽벽안쪽x좌표,
  "wall_right_x": 오른쪽벽안쪽x좌표,
  "game_score": 실제게임점수숫자,
  "ceiling_y": 천장흰점선y좌표,
  "game_area": {"width": 벽사이너비, "height": 게임영역높이}
}
```

**중요 규칙**:
- 좌표는 이미지 왼쪽 상단 (0,0) 기준 픽셀 좌표
- current_coin은 반드시 감지해야 한다 (게임 중이면 항상 존재)
- coins 배열이 비어있어도 current_coin과 next_coin은 반환해야 한다
- 벽 위치(wall_left_x, wall_right_x)를 정확히 감지해야 한다
- game_score는 상단 바의 노란 동전 옆 숫자를 정확히 읽어야 한다
- ceiling_y는 흰색 점선의 y좌표를 정확히 감지해야 한다
- JSON만 출력하고 다른 설명은 포함하지 않는다
"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash", max_calls_per_minute: int = 10):
        """
        Args:
            api_key: Google Gemini API 키
            model_name: 사용할 모델명
        """
        self.api_key = api_key
        self.model_name = model_name
        self.max_calls_per_minute = max_calls_per_minute
        self._call_timestamps = deque()
        
        if not api_key:
            raise ValueError("GEMINI_API_KEY가 설정되지 않았습니다. 환경 변수를 확인하세요.")
 
        self._client = None
        self._model = None
 
        # Gemini API 초기화 (신형 SDK 우선)
        if _HAS_NEW_SDK:
            self._client = genai.Client(api_key=api_key)
            self._model = model_name
        elif _HAS_LEGACY_SDK:
            legacy_genai.configure(api_key=api_key)
            self._client = legacy_genai
            self._model = legacy_genai.GenerativeModel(model_name)
        else:
            raise ImportError("Gemini SDK가 설치되지 않았습니다. 'google-genai' 또는 'google-generativeai'를 설치하세요.")
        
        logger.info(f"GeminiAnalyzer 초기화: 모델 = {model_name}")

    def _rate_limit(self):
        now = time.time()
        window_start = now - 60.0
        while self._call_timestamps and self._call_timestamps[0] < window_start:
            self._call_timestamps.popleft()
 
        if len(self._call_timestamps) >= self.max_calls_per_minute:
            sleep_s = (self._call_timestamps[0] + 60.0) - now
            if sleep_s > 0:
                time.sleep(sleep_s)
 
        self._call_timestamps.append(time.time())
    
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

            self._rate_limit()
             
            # 프롬프트 구성
            prompt = self.SYSTEM_PROMPT + "\n\n이미지를 분석하고 JSON 형식으로 결과를 반환하라."
             
            # Gemini API 호출
            if _HAS_NEW_SDK and self._client is not None:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=[prompt, image],
                )
                response_text = (response.text or "").strip()
            else:
                response = self._model.generate_content([prompt, image])
                response_text = response.text.strip()
             
            # 응답 텍스트 추출
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
    
    def get_current_coin_type(self, analysis_result: Dict) -> Optional[CoinType]:
        """
        분석 결과에서 현재 떨어뜨릴 동전 타입 추출
        
        Args:
            analysis_result: analyze_image()의 반환값
            
        Returns:
            현재 동전의 CoinType 또는 None
        """
        name = analysis_result.get('current_coin')
        if not name:
            return None
        return CoinType.from_name(name)

    def analyze_and_extract(self, image: Image.Image) -> tuple[List[Coin], Optional[CoinType], Optional[CoinType], Dict]:
        """
        이미지 분석 및 동전 추출을 한 번에 수행
        
        Args:
            image: PIL Image 객체
            
        Returns:
            (동전 리스트, 현재 동전 타입, 다음 동전 타입, 게임 영역 정보)
            게임 영역 정보에 wall_left_x, wall_right_x 포함
        """
        # 이미지 분석
        result = self.analyze_image(image)
        
        if not result:
            return [], None, None, {}
        
        # 동전 추출
        coins = self.extract_coins(result)
        
        # 현재 동전 타입 (지금 떨어뜨릴 동전)
        current_coin = self.get_current_coin_type(result)
        
        # 다음 동전 타입
        next_coin = self.get_next_coin_type(result)
        
        # 게임 영역 (벽 위치, 점수, 천장 포함)
        game_area = result.get('game_area', {})
        game_area['wall_left_x'] = result.get('wall_left_x')
        game_area['wall_right_x'] = result.get('wall_right_x')
        game_area['game_score'] = result.get('game_score')
        game_area['ceiling_y'] = result.get('ceiling_y')
        
        return coins, current_coin, next_coin, game_area


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
