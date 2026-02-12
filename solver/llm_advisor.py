"""
LLM-Only 동전게임 전문가 어드바이저

물리 시뮬레이터 없이 Gemini Flash 3.0 Preview가 직접 게임 화면을 보고
최적 드롭 위치를 판단합니다. 이전/현재 상태를 비교하여 자체 피드백하고 개선합니다.
"""

import json
import logging
import time
from collections import deque
from typing import Optional, Dict, List, Tuple
from PIL import Image

try:
    from google import genai
    from google.genai import types as genai_types
    _HAS_SDK = True
except Exception as _sdk_err:
    genai = None
    genai_types = None
    _HAS_SDK = False
    import traceback as _tb
    print(f"[WARN] google.genai import failed: {_sdk_err}")
    _tb.print_exc()

logger = logging.getLogger(__name__)


# ============================================================
# 수박게임(동전게임) 전문가 시스템 프롬프트
# ============================================================

EXPERT_SYSTEM_PROMPT = """당신은 '콘텐츠페이 동전게임'(수박게임 변형) 전문가 AI입니다.
게임 화면 이미지를 보고 최적의 드롭 위치를 판단합니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ 절대 규칙 #1: 낙하 경로 검증 (이것을 어기면 0점)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

동전은 위에서 아래로 **수직 낙하**한다.
떨어뜨린 동전은 **처음 만나는 장애물(다른 동전 또는 바닥) 위에 멈춘다.**

따라서:
  ✗ 목표 동전 위에 다른 동전이 쌓여있으면 → 절대 합성 불가!
  ✗ 떨어뜨린 동전은 가로막는 동전 위에 멈추고 목표까지 도달 못 함!
  ✓ 합성하려면 drop_x에서 수직 아래로 내려갈 때 목표 동전이 "처음 만나는 동전"이어야 함!

[구체적 예시]
  상황: 바닥에 핑크동전(y=800), 그 위에 주황동전(y=730)이 쌓여있음
  → 핑크동전을 x=같은위치에 떨어뜨리면?
  → 주황동전(y=730)에 먼저 부딪혀서 주황동전 위에 멈춤!
  → 핑크동전끼리 합성 안 됨! 완전히 잘못된 수!

  상황: 왼쪽에 노랑동전(y=750), 그 위에 검정번개(y=700), 그 위에 핑크동전(y=660)
  → 노랑동전을 떨어뜨리면? → 핑크동전(y=660) 위에 멈춤. 노랑끼리 합성 불가!

[경로 검증 방법 — 반드시 수행할 것]
  1. drop_x 위치에서 y=0(화면 상단)부터 아래로 스캔
  2. 그 x좌표 근처(동전 반지름 고려)에 있는 모든 동전을 y값 오름차순으로 나열
  3. 가장 먼저 만나는(y값이 가장 작은) 동전이 합성 대상인가?
  4. 아니면 → 그 합성은 불가능. 다른 위치를 찾아라.

★ 이 검증을 건너뛰고 합성을 추천하면 완전히 틀린 답이다!
★ JSON 응답에 반드시 "path_check" 필드로 경로 검증 결과를 기록하라!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ 좌표계
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- (0,0) = 화면 왼쪽 상단. x: 왼→오 증가. y: 위→아래 증가.
- y가 작을수록 위쪽, 클수록 아래쪽.
- 동전이 쌓이면 y값이 작아진다(위로 올라감).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ 점수 표시 (혼동 주의!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

화면 상단에 숫자 2개가 있다:
- 🏆 트로피 아이콘 옆 = **최고 점수** (역대 기록, 참고용)
- 🟡 노란 동그라미 옆 = **현재 점수** (이번 게임 점수)
→ game_score에는 반드시 **현재 점수**(노란 동그라미 옆)를 넣어라!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ 게임 규칙
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 상단에서 동전을 좌우 이동 후 떨어뜨린다.
2. 같은 종류 동전 2개가 접촉하면 합성 → 한 단계 위 동전 생성.
3. 합성 시 두 동전의 중간 위치에 새 동전 생성, 주변 밀려남 가능.
4. 동전이 천장 라인(흰색 점선) 넘으면 게임 오버.
5. 랜덤 동전은 레벨 1~6까지만 나옴.
6. 목표: 최대한 높은 점수.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ 물리 법칙 (미끄러짐/튕김 — 매우 중요!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

동전은 단순히 수직 낙하하지 않는다! 다음 물리를 반드시 고려하라:

1. **미끄러짐**: 동전이 다른 동전 위에 떨어지면 둥근 표면을 따라 좌우로 미끄러져 굴러간다.
   - 동전 위의 경사면에 떨어지면 낮은 쪽으로 굴러감
   - 따라서 정확히 위에 떨어뜨려도 옆으로 굴러갈 수 있음
   - 특히 작은 동전(Lv1~3)은 크게 미끄러짐

2. **튕김**: 동전은 탄성이 있어 충돌 시 통통 튀다.
   - 높은 곳에서 떨어질수록 더 많이 튀
   - 튕김으로 예상치 못한 위치로 이동할 수 있음
   - 합성 후 생성된 동전도 주변 동전을 밀어내며 튕김 유발

3. **낙하 예측 보정**:
   - 동전 위에 떨어뜨릴 때: 목표 동전 정중앙보다 약간 안쪽(벽 방향)으로 보정
   - 동전 사이 틈새로 떨어뜨릴 때: 틈새 너비가 동전 지름보다 충분히 큰지 확인
   - 이전 턴에서 예상과 다른 결과가 나왔다면 → 미끄러짐을 고려한 보정이 필요

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ 게임오버 방지 (절대 규칙 #2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

동전이 천장 라인을 넘으면 즉시 게임오버다. 이것을 방지하는 것이 합성보다 우선이다!

- 가장 높은 동전의 y좌표와 천장 y좌표를 비교하라
- 가장 높은 동전의 y가 천장+동전반지름 이하면 → 위험! (danger)
- 위험 상황에서는:
  ✓ 높은 곳에 동전을 쌓지 마라
  ✓ 가장 낮은 지점에 떨어뜨려라
  ✓ 합성으로 높이를 줄일 수 있는 수를 최우선 선택
  ✓ 동전을 떨어뜨렸을 때 튕김/미끄러짐으로 동전이 위로 올라갈 수 있음을 고려
  ✓ 특히 동전 더미 위에 떨어뜨릴 때 튕김으로 위로 튜어오를 수 있음에 주의

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ 동전 계층 (11단계)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Lv  이름        반지름  점수   색상/특징
 1  검정번개     18px    10   매우 작고 어두운, 번개 무늬
 2  핑크동전     27px    20   작은 분홍, 하트 무늬
 3  주황동전     35px    40   중소 주황색
 4  노랑동전     42px    80   중형 노랑, 별 무늬
 5  민트동전     48px   160   중형 민트/연두, 클로버
 6  파랑동전     55px   320   중대형 파랑, 다이아몬드
 7  보라동전     62px   640   대형 보라
 8  갈색동전     70px  1280   대형 갈색
 9  흰색상자     78px  2560   매우 큰 흰색 사각형
10  노랑전구     87px  5120   매우 큰 노랑 전구
11  민트선물상자  96px 10240   가장 큰 민트색 선물상자

같은 레벨끼리만 합성. 민트선물상자×2 → 소멸(보너스 20480점).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ 판단 프로세스 (매 턴 반드시 이 순서대로)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1: 보드 스캔
  - 모든 동전의 종류(색상+크기), 중심 좌표(x,y) 식별
  - 현재 동전, 다음 동전 확인
  - 높이 위험도 평가

Step 2: 합성 후보 나열
  - 현재 동전과 같은 종류가 보드에 있는가?
  - 각 후보의 위치(x,y) 기록

Step 3: ★★★ 경로 검증 (가장 중요!) ★★★
  - 각 합성 후보에 대해:
    "drop_x에서 수직 낙하 시, 목표 동전이 처음 만나는 동전인가?"
  - 목표 동전 위에 다른 동전이 있으면 → 합성 불가! 후보에서 제외!
  - 경로가 열린 후보만 남긴다.

Step 4: 최적 위치 결정
  - 경로가 열린 합성 가능 후보 중 최선 선택
  - 합성 가능한 후보가 없으면 → 전략적 배치 (같은 종류 근처, 낮은 곳)
  - 다음 동전도 고려하여 2수 앞 계획

Step 5: 이전 턴 비교 (이전 화면 제공 시)
  - 직전 조언의 결과 평가, 같은 실수 반복 금지

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ 전략 요약
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- 큰 동전은 한쪽 벽에 고정, 위로 갈수록 작은 동전 (피라미드)
- 같은 종류끼리 가까이 배치 → 합성 기회 극대화
- 작은 동전(1~3)은 빨리 합성하여 공간 확보
- 연쇄 합성(A+A→B, B+B→C) 유도
- 높이 위험 시 → 합성 가능한 곳 우선, 없으면 가장 낮은 곳
- 현재 동전 + 다음 동전 모두 고려 (2수 앞)
- 미끄러짐 보정: 동전 위에 떨어뜨릴 때 약간 안쪽(벽 방향)으로 보정하여 굴러감 방지
- 게임오버 방지가 합성보다 우선! 천장 근처에 동전이 있으면 절대 그 위에 쌓지 마라
"""

ANALYSIS_PROMPT_TEMPLATE = """이 게임 화면을 분석하고 최적의 드롭 위치를 결정하라.

■ 분석 순서 (반드시 이 순서대로):
1. 보드 위의 모든 동전을 식별하라 (종류, x, y).
2. 현재 동전과 같은 종류가 보드에 있는지 찾아라.
3. ★★★ 각 합성 후보에 대해 낙하 경로를 검증하라! ★★★
   - drop_x에서 수직 아래로 내려갈 때, 목표 동전 위에 다른 동전이 있으면 합성 불가!
   - 경로가 막힌 후보는 반드시 제외하라!
4. 경로가 열린 합성 후보 중 최선을 선택하라.
5. 합성 가능한 후보가 없으면 전략적 배치 (낮은 곳, 같은 종류 근처).
6. 점수: 🏆트로피 옆 = 최고점수(참고용), 🟡노란동그라미 옆 = 현재점수.

{feedback_section}

{bounds_section}

■ 출력 (반드시 JSON만):
```json
{{
  "current_coin": "현재동전종류",
  "next_coin": "다음동전종류",
  "coins": [
    {{"type": "동전종류", "x": 중심x, "y": 중심y, "radius": 반지름}}
  ],
  "highest_coin_y": 가장높은동전의y좌표,
  "game_score": 현재점수숫자,
  "drop_x": 최적드롭x좌표,
  "confidence": 0.0~1.0,
  "path_check": "drop_x에서 수직 낙하 시 처음 만나는 동전이 무엇인지, 합성 대상에 도달 가능한지 설명",
  "sliding_risk": "미끄러짐/튕김 위험 평가 — 낙하 후 동전이 어디로 굴러갈지 예측",
  "reason": "한글 2~3줄 이유 (미끄러짐/게임오버 위험 포함)",
  "strategy": "전략 이름",
  "risk_level": "safe/warning/danger",
  "alternative_x": 차선책x좌표,
  "alternative_reason": "차선책 이유"
}}
```

■ 핵심 규칙:
- path_check 필드를 반드시 작성하라! 경로 검증 없이 합성을 추천하면 틀린 답이다.
- sliding_risk 필드를 반드시 작성하라! 미끄러짐/튕김 예측 없이 추천하면 불완전한 답이다.
- highest_coin_y가 천장 근처이면 반드시 risk_level을 danger로 설정하라.
- drop_x는 벽 안쪽 범위 내의 픽셀 x좌표
- reason은 한글로 핵심만 2~3줄
- JSON만 출력, 다른 텍스트 금지
"""


class LLMAdvisor:
    """LLM 기반 동전게임 전문가 어드바이저"""

    def __init__(self, api_key: str, model_name: str = "gemini-3-flash-preview"):
        if not _HAS_SDK:
            raise ImportError("google-genai SDK가 필요합니다. pip install google-genai")
        if not api_key:
            raise ValueError("API 키가 필요합니다.")

        self.api_key = api_key
        self.model_name = model_name
        self._client = genai.Client(api_key=api_key)

        # 이전 상태 기록 (피드백 루프용)
        self._history: deque = deque(maxlen=10)  # 최근 10턴 기록
        self._last_advice: Optional[Dict] = None
        self._last_image: Optional[Image.Image] = None
        self._last_board_state: Optional[Dict] = None

        # 레이트 리밋
        self._call_timestamps: deque = deque()
        self._max_calls_per_minute = 15

        # 경계값
        self.wall_left = 78.0
        self.wall_right = 468.0
        self.ceiling_y = 380.0
        self.floor_y = 930.0

        logger.info(f"LLMAdvisor 초기화: model={model_name}")

    def _rate_limit(self):
        now = time.time()
        window_start = now - 60.0
        while self._call_timestamps and self._call_timestamps[0] < window_start:
            self._call_timestamps.popleft()
        if len(self._call_timestamps) >= self._max_calls_per_minute:
            sleep_s = (self._call_timestamps[0] + 60.0) - now
            if sleep_s > 0:
                logger.info(f"Rate limit: {sleep_s:.1f}s 대기")
                time.sleep(sleep_s)
        self._call_timestamps.append(time.time())

    def _build_feedback_section(self) -> str:
        """이전 턴 피드백 섹션 생성"""
        if not self._history:
            return ""

        lines = ["■ 이전 턴 피드백 (자체 학습용):"]
        for i, h in enumerate(self._history):
            turn = len(self._history) - i
            lines.append(f"  [{turn}턴 전] 동전={h.get('current_coin','?')}, "
                         f"드롭x={h.get('drop_x','?')}, "
                         f"전략={h.get('strategy','?')}, "
                         f"결과={'성공' if h.get('success') else '개선필요'}")
            if h.get('feedback'):
                lines.append(f"    → 피드백: {h['feedback']}")

        if self._last_advice:
            lines.append(f"\n  [직전 조언] drop_x={self._last_advice.get('drop_x')}, "
                         f"이유: {self._last_advice.get('reason','')}")
            lines.append("  → 이번 화면을 보고 직전 조언이 좋았는지 스스로 평가하고 개선하라.")

        return "\n".join(lines)

    def _build_bounds_section(self) -> str:
        """경계값 섹션 생성"""
        danger_y = self.ceiling_y + 50
        warning_y = self.ceiling_y + 100
        return (f"■ 현재 경계값 (사용자가 설정한 정확한 값):\n"
                f"  벽 왼쪽: {self.wall_left:.0f}px\n"
                f"  벽 오른쪽: {self.wall_right:.0f}px\n"
                f"  천장: {self.ceiling_y:.0f}px\n"
                f"  바닥: {self.floor_y:.0f}px\n"
                f"  → drop_x는 반드시 {self.wall_left:.0f} ~ {self.wall_right:.0f} 범위 안이어야 한다.\n"
                f"  → 🚨 위험 구간: 동전 y < {danger_y:.0f}px 이면 게임오버 임박!\n"
                f"  → ⚠ 주의 구간: 동전 y < {warning_y:.0f}px 이면 높이 위험")

    def analyze(self, image: Image.Image,
                progress_callback=None) -> Optional[Dict]:
        """
        게임 화면을 분석하고 최적 드롭 위치를 반환.

        Returns:
            {
                "drop_x": float,
                "confidence": float,
                "reason": str,
                "strategy": str,
                "risk_level": str,
                "current_coin": str,
                "next_coin": str,
                "coins": [...],
                "highest_coin_y": float,
                "sliding_risk": str,
                "game_score": int,
                "alternative_x": float,
                "alternative_reason": str,
            }
        """
        try:
            if progress_callback:
                progress_callback(0.1, "LLM 분석 중...")

            self._rate_limit()

            feedback_section = self._build_feedback_section()
            bounds_section = self._build_bounds_section()

            prompt = ANALYSIS_PROMPT_TEMPLATE.format(
                feedback_section=feedback_section,
                bounds_section=bounds_section,
            )

            if progress_callback:
                progress_callback(0.2, "Gemini API 호출 중...")

            # contents 구성: google.genai SDK는 [str, Image, str, ...] 리스트 형태만 지원
            contents = []
            if self._last_image is not None and self._last_advice is not None:
                contents.append("[이전 화면 — 아래 이미지는 직전 턴의 게임 상태입니다]")
                contents.append(self._last_image)
                contents.append(f"[이전 조언: drop_x={self._last_advice.get('drop_x')}, "
                                f"전략={self._last_advice.get('strategy','?')}]")
                contents.append("\n[현재 화면 — 아래 이미지가 지금 분석할 게임 상태입니다]")
            contents.append(image)
            contents.append(EXPERT_SYSTEM_PROMPT + "\n\n" + prompt)

            # Gemini 3 Flash Preview + thinking_level=medium (딥 추론)
            t0 = time.time()
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    thinking_config=genai_types.ThinkingConfig(
                        thinking_level="medium"
                    ),
                    http_options={"timeout": 90_000},  # 90초 타임아웃
                ),
            )
            elapsed = time.time() - t0
            # contents 메모리 해제 (이미지 참조 제거)
            contents.clear()

            # 응답 텍스트 추출 (thinking 모드에서는 .text가 None일 수 있음)
            response_text = ""
            try:
                response_text = (response.text or "").strip()
            except Exception:
                # 폴백: candidates에서 직접 추출
                if response.candidates:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'text') and part.text:
                            response_text += part.text
                    response_text = response_text.strip()
            logger.debug(f"LLM 응답 ({elapsed:.1f}s): {response_text[:200]}...")

            if progress_callback:
                progress_callback(0.8, f"응답 파싱 중... ({elapsed:.1f}s)")

            result = self._parse_response(response_text)
            if not result:
                logger.warning("LLM 응답 파싱 실패")
                return None

            # drop_x 범위 클램핑
            drop_x = result.get("drop_x")
            if drop_x is not None:
                drop_x = max(self.wall_left + 20, min(self.wall_right - 20, float(drop_x)))
                result["drop_x"] = drop_x

            alt_x = result.get("alternative_x")
            if alt_x is not None:
                alt_x = max(self.wall_left + 20, min(self.wall_right - 20, float(alt_x)))
                result["alternative_x"] = alt_x

            # 피드백 루프: 이전 조언 평가
            self._evaluate_previous(result)

            # 현재 조언 저장 (이미지는 복사본 저장 — 원본은 호출자가 해제)
            self._last_advice = result
            self._last_image = image.copy() if image else None

            if progress_callback:
                progress_callback(1.0, f"완료 ({elapsed:.1f}s)")

            logger.info(f"LLM 조언: drop_x={result.get('drop_x')}, "
                        f"conf={result.get('confidence')}, "
                        f"전략={result.get('strategy')}, "
                        f"이유={result.get('reason','')[:50]}")

            return result

        except Exception as e:
            logger.error(f"LLM 분석 실패: {e}", exc_info=True)
            if progress_callback:
                progress_callback(0.0, f"분석 실패: {e}")
            return None

    def _evaluate_previous(self, current_result: Dict):
        """이전 조언을 현재 상태와 비교하여 피드백 생성"""
        if not self._last_advice:
            return

        prev = self._last_advice
        curr_coins = current_result.get("coins", [])
        prev_coins = prev.get("coins", [])
        prev_coins_count = len(prev_coins)
        curr_coins_count = len(curr_coins)

        prev_score = prev.get("game_score") or 0
        curr_score = current_result.get("game_score") or 0

        success = True
        feedback_parts = []

        # 1. 점수 변화 분석
        if isinstance(curr_score, (int, float)) and isinstance(prev_score, (int, float)):
            score_diff = curr_score - prev_score
            if score_diff > 0:
                feedback_parts.append(f"점수 +{score_diff} 상승")
            elif score_diff == 0:
                feedback_parts.append("점수 변화 없음")
                success = False

        # 2. 동전 수 변화 (합성 감지)
        coin_diff = prev_coins_count - curr_coins_count
        if coin_diff > 0:
            # 동전이 줄었다 = 합성 발생 (떨어뜨린 1개 추가 - 합성으로 사라진 수)
            merges = (coin_diff + 1) // 2  # 대략적 합성 횟수
            feedback_parts.append(f"합성 {merges}회 추정 (동전 {prev_coins_count}→{curr_coins_count})")
            success = True
        elif coin_diff == -1:
            feedback_parts.append("합성 없이 동전 1개 추가됨")
            success = False
        elif coin_diff < -1:
            feedback_parts.append(f"동전 급증 ({prev_coins_count}→{curr_coins_count})")

        # 3. 높이 위험도 분석
        if curr_coins:
            valid_ys = [c.get("y", 999) for c in curr_coins if isinstance(c, dict)]
            if valid_ys:
                min_y = min(valid_ys)
                height_pct = max(0, (self.floor_y - min_y) / max(1, self.floor_y - self.ceiling_y))
                if height_pct > 0.85:
                    feedback_parts.append(f"🚨 높이 {height_pct*100:.0f}% — 매우 위험!")
                    success = False
                elif height_pct > 0.7:
                    feedback_parts.append(f"⚠ 높이 {height_pct*100:.0f}% — 주의")

        # 4. 전략 평가
        prev_strategy = prev.get("strategy", "")
        if "합성" in prev_strategy and coin_diff <= 0:
            feedback_parts.append(f"전략 '{prev_strategy}' 실패 — 합성 미발생")
            success = False
        elif "합성" in prev_strategy and coin_diff > 0:
            feedback_parts.append(f"전략 '{prev_strategy}' 성공")

        feedback = ". ".join(feedback_parts) if feedback_parts else "평가 불가"

        self._history.append({
            "current_coin": prev.get("current_coin"),
            "drop_x": prev.get("drop_x"),
            "strategy": prev.get("strategy"),
            "success": success,
            "feedback": feedback,
            "score_before": prev_score,
            "score_after": curr_score,
            "coins_before": prev_coins_count,
            "coins_after": curr_coins_count,
        })

    def _parse_response(self, text: str) -> Optional[Dict]:
        """JSON 응답 파싱"""
        try:
            # ```json ... ``` 블록 추출
            if "```json" in text:
                start = text.find("```json") + 7
                end = text.find("```", start)
                text = text[start:end].strip()
            elif "```" in text:
                start = text.find("```") + 3
                end = text.find("```", start)
                text = text[start:end].strip()

            # 직접 파싱 시도
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                # 폴백: 텍스트에서 첫 번째 { ... } 블록 추출
                brace_start = text.find("{")
                brace_end = text.rfind("}")
                if brace_start >= 0 and brace_end > brace_start:
                    text = text[brace_start:brace_end + 1]
                    data = json.loads(text)
                else:
                    raise

            # 필수 필드 검증
            if "drop_x" not in data:
                logger.warning("응답에 drop_x 없음")
                return None

            # 기본값 설정
            data.setdefault("confidence", 0.5)
            data.setdefault("reason", "분석 완료")
            data.setdefault("strategy", "기본")
            data.setdefault("risk_level", "safe")
            data.setdefault("coins", [])
            data.setdefault("alternative_x", data["drop_x"])
            data.setdefault("alternative_reason", "")

            return data

        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 오류: {e}")
            logger.debug(f"원본: {text[:300]}")
            return None
        except Exception as e:
            logger.error(f"파싱 오류: {e}")
            return None

    def chat(self, user_message: str, image: Optional[Image.Image] = None) -> str:
        """사용자와 대화 — 게임 컨텍스트를 포함하여 LLM에게 질문/피드백 전달.

        Args:
            user_message: 사용자가 입력한 메시지
            image: 현재 게임 화면 (없으면 마지막 저장된 이미지 사용)

        Returns:
            LLM의 한글 응답 텍스트
        """
        try:
            self._rate_limit()

            img = image or self._last_image

            chat_prompt = (
                "당신은 동전게임(수박게임 변형) 전문가 AI 어시스턴트입니다.\n"
                "사용자가 게임에 대해 질문하거나 피드백을 줍니다.\n"
                "게임 화면 이미지가 제공되면 현재 상태를 참고하여 답하세요.\n"
                "반드시 한글로 답하세요. 간결하고 실용적으로 답하세요.\n\n"
                "핵심 규칙 리마인더:\n"
                "- 동전은 수직 낙하하며 처음 만나는 장애물 위에 멈춤\n"
                "- 목표 동전 위에 다른 동전이 있으면 합성 불가\n"
                "- 🏆트로피 옆 = 최고점수, 🟡노란동그라미 옆 = 현재점수\n\n"
            )

            # 최근 조언 컨텍스트 추가
            if self._last_advice:
                adv = self._last_advice
                chat_prompt += (
                    f"[최근 분석 결과]\n"
                    f"  드롭 위치: x={adv.get('drop_x')}\n"
                    f"  전략: {adv.get('strategy','?')}\n"
                    f"  이유: {adv.get('reason','?')}\n"
                    f"  경로 검증: {adv.get('path_check','미확인')}\n\n"
                )

            chat_prompt += f"사용자 메시지: {user_message}"

            contents = []
            if img is not None:
                contents.append("[현재 게임 화면]")
                contents.append(img)
            contents.append(chat_prompt)

            response = self._client.models.generate_content(
                model=self.model_name,
                contents=contents,
            )

            response_text = ""
            try:
                response_text = (response.text or "").strip()
            except Exception:
                if response.candidates:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'text') and part.text:
                            response_text += part.text
                    response_text = response_text.strip()

            if not response_text:
                response_text = "(응답 없음)"

            logger.info(f"Chat 응답: {response_text[:100]}...")
            return response_text

        except Exception as e:
            logger.error(f"Chat 실패: {e}", exc_info=True)
            return f"오류 발생: {e}"

    def get_history_summary(self) -> str:
        """최근 히스토리 요약"""
        if not self._history:
            return "기록 없음"
        total = len(self._history)
        success = sum(1 for h in self._history if h.get("success"))
        return f"최근 {total}턴: 성공 {success}/{total} ({success/total*100:.0f}%)"
