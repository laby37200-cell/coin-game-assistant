"""
Pymunk 물리 엔진 래퍼

게임 상태를 물리 공간에 복제하고 시뮬레이션을 실행합니다.
"""

import pymunk
from typing import List, Tuple, Optional, Deque
import logging
from collections import deque

from models.coin import Coin, CoinType


logger = logging.getLogger(__name__)


class PhysicsEngine:
    """Pymunk 물리 엔진 래퍼 클래스"""
    
    def __init__(
        self,
        game_width: int,
        game_height: int,
        gravity: Tuple[float, float] = (0, -900),
        damping: float = 0.95,
        iterations: int = 10,
        coin_friction: float = 0.5,
        coin_elasticity: float = 0.3,
        wall_friction: float = 0.6,
        wall_elasticity: float = 0.2
    ):
        """
        Args:
            game_width: 게임 영역 너비 (픽셀)
            game_height: 게임 영역 높이 (픽셀)
            gravity: 중력 벡터 (x, y)
            damping: 감쇠 계수 (0~1)
            iterations: 물리 시뮬레이션 반복 횟수
        """
        self.game_width = game_width
        self.game_height = game_height
        self.gravity = gravity
        self.damping = damping
        self.iterations = iterations
        self.coin_friction = coin_friction
        self.coin_elasticity = coin_elasticity
        self.wall_friction = wall_friction
        self.wall_elasticity = wall_elasticity
        
        # Pymunk Space 생성
        self.space = pymunk.Space()
        self.space.gravity = self.gravity
        self.space.damping = self.damping
        self.space.iterations = self.iterations
        
        # 바디 추적용 딕셔너리
        self.coin_bodies = {}  # {coin_id: (body, shape)}
        self.wall_bodies = []

        # 합체(merge) 처리
        self._pending_merges: Deque[Tuple[int, int]] = deque()
        self._merged_body_ids_in_step: set[Tuple[int, int]] = set()
        self._merge_score_accum: float = 0.0
        
        logger.info(f"PhysicsEngine 초기화: {game_width}x{game_height}, "
                   f"중력={gravity}, 감쇠={damping}")
        
        # 벽과 바닥 생성
        self._create_walls()

        # 충돌 핸들러 설정 (동일 동전 합체)
        self._setup_collision_handlers()
    
    def _create_walls(self):
        """게임 영역의 벽과 바닥 생성"""
        # 정적 바디 (움직이지 않음)
        static_body = self.space.static_body
        
        # 벽 두께
        wall_thickness = 10
        
        # 바닥 (y = 0)  - y-up 좌표계
        floor = pymunk.Segment(
            static_body,
            (0, 0),
            (self.game_width, 0),
            wall_thickness
        )
        floor.friction = self.wall_friction
        floor.elasticity = self.wall_elasticity
        
        # 왼쪽 벽 (x = 0)
        left_wall = pymunk.Segment(
            static_body,
            (0, 0),
            (0, self.game_height),
            wall_thickness
        )
        left_wall.friction = self.wall_friction
        left_wall.elasticity = self.wall_elasticity
        
        # 오른쪽 벽 (x = game_width)
        right_wall = pymunk.Segment(
            static_body,
            (self.game_width, 0),
            (self.game_width, self.game_height),
            wall_thickness
        )
        right_wall.friction = self.wall_friction
        right_wall.elasticity = self.wall_elasticity
        
        # Space에 추가
        self.space.add(floor, left_wall, right_wall)
        self.wall_bodies = [floor, left_wall, right_wall]
        
        logger.debug("벽과 바닥 생성 완료")
    
    def add_coin(
        self,
        coin: Coin,
        is_static: bool = False,
        friction: Optional[float] = None,
        elasticity: Optional[float] = None
    ) -> int:
        """
        동전을 물리 공간에 추가
        
        Args:
            coin: 추가할 동전
            is_static: 정적 바디 여부 (움직이지 않음)
            friction: 마찰 계수
            elasticity: 탄성 계수
            
        Returns:
            동전 ID
        """
        # 바디 생성
        if is_static:
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
        else:
            mass = coin.mass
            moment = pymunk.moment_for_circle(mass, 0, coin.radius)
            body = pymunk.Body(mass, moment)
        
        # 위치 설정
        body.position = (coin.x, coin.y)
        
        # 속도 설정
        body.velocity = (coin.velocity_x, coin.velocity_y)
        
        # 원형 Shape 생성
        shape = pymunk.Circle(body, coin.radius)
        shape.friction = self.coin_friction if friction is None else friction
        shape.elasticity = self.coin_elasticity if elasticity is None else elasticity

        # 동전 타입 정보 저장 (충돌 감지용)
        shape.coin_type = coin.coin_type
        shape.collision_type = 1
        
        # Space에 추가
        self.space.add(body, shape)
        
        # ID 생성 및 저장
        coin_id = id(coin)
        self.coin_bodies[coin_id] = (body, shape)
        
        logger.debug(f"동전 추가: {coin.coin_type.display_name} at ({coin.x:.1f}, {coin.y:.1f})")
        
        return coin_id

    def _setup_collision_handlers(self):
        # pymunk 7.x uses on_collision() with keyword callbacks
        self.space.on_collision(
            collision_type_a=1,
            collision_type_b=1,
            begin=self._on_coin_contact_begin,
        )

    def _on_coin_contact_begin(self, arbiter: pymunk.Arbiter, space: pymunk.Space, data) -> None:
        try:
            shape_a, shape_b = arbiter.shapes
            if getattr(shape_a, "coin_type", None) is None or getattr(shape_b, "coin_type", None) is None:
                return

            if shape_a.coin_type != shape_b.coin_type:
                return

            body_a_id = id(shape_a.body)
            body_b_id = id(shape_b.body)
            key = (body_a_id, body_b_id) if body_a_id < body_b_id else (body_b_id, body_a_id)
            if key in self._merged_body_ids_in_step:
                return

            self._merged_body_ids_in_step.add(key)
            self._pending_merges.append((body_a_id, body_b_id))
        except Exception:
            pass

    def process_pending_merges(self):
        if not self._pending_merges:
            self._merged_body_ids_in_step.clear()
            return

        body_by_id = {id(body): (coin_id, body, shape) for coin_id, (body, shape) in self.coin_bodies.items()}
        while self._pending_merges:
            a_id, b_id = self._pending_merges.popleft()
            if a_id not in body_by_id or b_id not in body_by_id:
                continue

            coin_id_a, body_a, shape_a = body_by_id[a_id]
            coin_id_b, body_b, shape_b = body_by_id[b_id]
            if shape_a.coin_type != shape_b.coin_type:
                continue

            coin_type: CoinType = shape_a.coin_type
            next_type = coin_type.get_next_level()

            # 제거
            self.space.remove(body_a, shape_a)
            self.space.remove(body_b, shape_b)
            if coin_id_a in self.coin_bodies:
                del self.coin_bodies[coin_id_a]
            if coin_id_b in self.coin_bodies:
                del self.coin_bodies[coin_id_b]

            # 점수 누적(합체로 얻는 점수는 next 코인의 점수로 가정)
            if next_type is not None:
                self._merge_score_accum += float(next_type.score)

                # 새 코인 생성
                new_x = (body_a.position.x + body_b.position.x) / 2
                new_y = (body_a.position.y + body_b.position.y) / 2
                new_coin = Coin(coin_type=next_type, x=new_x, y=new_y)
                new_coin_id = self.add_coin(new_coin, is_static=False)
                body_by_id[id(self.coin_bodies[new_coin_id][0])] = (new_coin_id, self.coin_bodies[new_coin_id][0], self.coin_bodies[new_coin_id][1])

        self._merged_body_ids_in_step.clear()

    def pop_merge_score(self) -> float:
        score = self._merge_score_accum
        self._merge_score_accum = 0.0
        return score
    
    def remove_coin(self, coin_id: int):
        """
        동전을 물리 공간에서 제거
        
        Args:
            coin_id: 제거할 동전 ID
        """
        if coin_id not in self.coin_bodies:
            logger.warning(f"존재하지 않는 동전 ID: {coin_id}")
            return
        
        body, shape = self.coin_bodies[coin_id]
        
        # Space에서 제거
        try:
            self.space.remove(shape, body)
        except Exception:
            try:
                self.space.remove(shape)
            except Exception:
                pass
        
        # 딕셔너리에서 제거
        del self.coin_bodies[coin_id]
        
        logger.debug(f"동전 제거: ID={coin_id}")
    
    def get_coin_position(self, coin_id: int) -> Optional[Tuple[float, float]]:
        """
        동전의 현재 위치 반환
        
        Args:
            coin_id: 동전 ID
            
        Returns:
            (x, y) 위치 또는 None
        """
        if coin_id not in self.coin_bodies:
            return None
        
        body, _ = self.coin_bodies[coin_id]
        return (body.position.x, body.position.y)
    
    def get_coin_velocity(self, coin_id: int) -> Optional[Tuple[float, float]]:
        """
        동전의 현재 속도 반환
        
        Args:
            coin_id: 동전 ID
            
        Returns:
            (vx, vy) 속도 또는 None
        """
        if coin_id not in self.coin_bodies:
            return None
        
        body, _ = self.coin_bodies[coin_id]
        return (body.velocity.x, body.velocity.y)
    
    def step(self, dt: float = 1/60):
        """
        물리 시뮬레이션 한 스텝 진행
        
        Args:
            dt: 타임스텝 (초)
        """
        self.space.step(dt)
    
    def is_stable(self, velocity_threshold: float = 0.1) -> bool:
        """
        모든 동전이 안정 상태인지 확인
        
        Args:
            velocity_threshold: 안정 상태 판단 속도 임계값
            
        Returns:
            안정 상태 여부
        """
        for coin_id, (body, _) in self.coin_bodies.items():
            if body.body_type == pymunk.Body.STATIC:
                continue
            
            # 속도 크기 계산
            velocity_magnitude = (body.velocity.x ** 2 + body.velocity.y ** 2) ** 0.5
            
            if velocity_magnitude > velocity_threshold:
                return False
        
        return True
    
    def get_all_coins_state(self) -> List[Coin]:
        """
        현재 물리 공간의 모든 동전 상태 반환
        
        Returns:
            Coin 객체 리스트
        """
        coins = []
        
        for coin_id, (body, shape) in self.coin_bodies.items():
            coin = Coin(
                coin_type=shape.coin_type,
                x=body.position.x,
                y=body.position.y,
                velocity_x=body.velocity.x,
                velocity_y=body.velocity.y
            )
            coins.append(coin)
        
        return coins
    
    def clear(self):
        """물리 공간 초기화 (모든 동전 제거)"""
        # 모든 동전 제거
        for coin_id in list(self.coin_bodies.keys()):
            self.remove_coin(coin_id)
        
        logger.debug("물리 공간 초기화 완료")
    
    def reset(self):
        """물리 엔진 완전 리셋 (벽 포함 재생성)"""
        # Space 재생성
        self.space = pymunk.Space()
        self.space.gravity = self.gravity
        self.space.damping = self.damping
        self.space.iterations = self.iterations
        
        # 바디 딕셔너리 초기화
        self.coin_bodies = {}
        self.wall_bodies = []
        
        # 벽 재생성
        self._create_walls()

        # 충돌 핸들러 재설정
        self._setup_collision_handlers()
         
        logger.debug("물리 엔진 리셋 완료")


# 테스트 코드
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== Pymunk 물리 엔진 테스트 ===\n")
    
    # PhysicsEngine 생성
    engine = PhysicsEngine(game_width=600, game_height=800)
    print("✅ 물리 엔진 초기화 완료")
    
    # 테스트 동전 생성
    test_coin = Coin(
        coin_type=CoinType.YELLOW_CIRCLE,
        x=300,
        y=100
    )
    
    # 동전 추가
    coin_id = engine.add_coin(test_coin)
    print(f"✅ 동전 추가: ID={coin_id}")
    
    # 시뮬레이션 실행
    print("\n시뮬레이션 실행 중...")
    for i in range(100):
        engine.step(1/60)
        
        if i % 20 == 0:
            pos = engine.get_coin_position(coin_id)
            vel = engine.get_coin_velocity(coin_id)
            print(f"  Step {i:3d}: 위치=({pos[0]:.1f}, {pos[1]:.1f}), "
                  f"속도=({vel[0]:.1f}, {vel[1]:.1f})")
    
    # 안정 상태 확인
    if engine.is_stable():
        print("\n✅ 안정 상태 도달")
    else:
        print("\n⏳ 아직 움직이는 중...")
    
    # 최종 위치
    final_pos = engine.get_coin_position(coin_id)
    print(f"\n최종 위치: ({final_pos[0]:.1f}, {final_pos[1]:.1f})")
