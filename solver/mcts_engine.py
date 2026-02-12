"""
Monte Carlo Tree Search (MCTS) engine for coin game.

Inspired by AlphaZero/MuZero architecture:
  - Selection: UCB1 to balance exploration vs exploitation
  - Expansion: try all candidate drop positions
  - Simulation: GPU-accelerated physics rollout
  - Backpropagation: update node statistics

Supports 10-20 step lookahead with GPU batch simulation.
"""

import math
import time
import logging
import random
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field

from models.coin import Coin, CoinType
from solver.gpu_physics import GPUPhysicsBatch, PhysicsParams
from solver import fast_sim

logger = logging.getLogger(__name__)


@dataclass
class MCTSNode:
    """A node in the MCTS tree."""
    coins: List[Coin]           # game state at this node
    drop_coin: Optional[CoinType]  # coin to drop at this node (None = root before action)
    drop_x: Optional[float]    # x position chosen (None = root)
    parent: Optional['MCTSNode'] = None
    children: Dict[float, 'MCTSNode'] = field(default_factory=dict)  # x -> child node

    visits: int = 0
    total_value: float = 0.0
    merge_score: float = 0.0    # cumulative merge score from simulation

    @property
    def avg_value(self) -> float:
        return self.total_value / max(1, self.visits)

    def ucb1(self, exploration_weight: float = 1.414) -> float:
        """Upper Confidence Bound for tree selection."""
        if self.visits == 0:
            return float('inf')
        parent_visits = self.parent.visits if self.parent else self.visits
        exploit = self.avg_value
        explore = exploration_weight * math.sqrt(math.log(parent_visits + 1) / (self.visits + 1))
        return exploit + explore

    def best_child(self) -> Optional['MCTSNode']:
        """Select child with highest UCB1."""
        if not self.children:
            return None
        return max(self.children.values(), key=lambda c: c.ucb1())

    def best_action(self) -> Optional[float]:
        """Return the drop_x of the most visited child (robust action selection)."""
        if not self.children:
            return None
        best = max(self.children.values(), key=lambda c: c.visits)
        return best.drop_x


class MCTSEngine:
    """
    MCTS-based deep search engine for the coin game.

    Uses GPU-accelerated physics simulation for fast rollouts.
    Searches 10-20 steps deep depending on time budget.
    """

    def __init__(
        self,
        gpu_physics: GPUPhysicsBatch,
        wall_left: float = 105.0,
        wall_right: float = 435.0,
        floor_y: float = 870.0,
        ceiling_y: float = 200.0,
        num_positions: int = 12,
        max_depth: int = 15,
        time_budget_s: float = 8.0,
        max_iterations: int = 500,
    ):
        """
        Args:
            gpu_physics: GPU physics simulator
            wall_left/right: droppable area boundaries
            floor_y: floor y coordinate
            ceiling_y: ceiling (game over line) y coordinate
            num_positions: number of candidate drop positions to consider
            max_depth: maximum search depth (number of drops to look ahead)
            time_budget_s: time budget in seconds for search
            max_iterations: max MCTS iterations
        """
        self.gpu = gpu_physics
        self.wall_left = wall_left
        self.wall_right = wall_right
        self.floor_y = floor_y
        self.ceiling_y = ceiling_y
        self.num_positions = num_positions
        self.max_depth = max_depth
        self.time_budget_s = time_budget_s
        self.max_iterations = max_iterations

    def _candidate_positions(self, coin_type: CoinType) -> List[float]:
        """Generate candidate drop x positions within walls."""
        r = coin_type.radius
        lo = self.wall_left + r + 5
        hi = self.wall_right - r - 5
        if lo >= hi:
            return [(lo + hi) / 2]
        step = (hi - lo) / max(1, self.num_positions - 1)
        return [lo + i * step for i in range(self.num_positions)]

    def _random_next_coin(self) -> CoinType:
        """Sample a random droppable coin (level 1-6)."""
        return random.choice(CoinType.get_random_drop_coins())

    def _evaluate_state(self, coins: List[Coin], merge_score: float) -> float:
        """
        게임 상태 평가. 높을수록 좋음.

        핵심 게임 규칙:
          - 점수는 오직 합성(merge) 시에만 올라감
          - 상위 레벨 합성일수록 높은 점수
          - 천장 넘으면 게임오버 → 생존이 최우선
          - 같은 타입 코인이 가까이 있으면 곧 합성 가능 → 보너스
        """
        if not coins:
            return merge_score + 1000.0  # 전부 합성됨 = 최고

        value = merge_score

        # --- 게임오버 위험 페널티 (생존 최우선) ---
        for c in coins:
            top = c.y - c.radius
            margin = top - self.ceiling_y
            if margin < 0:
                value -= 10000 + abs(margin) * 200  # 게임오버 영역
            elif margin < 40:
                value -= (40 - margin) ** 2 * 2.0
            elif margin < 100:
                value -= (100 - margin) * 2.0

        # --- 코인 수 페널티 (적을수록 좋음 = 합성이 많이 됨) ---
        value -= len(coins) * 8.0

        # --- 합성 가능성 보너스 (같은 타입이 가까이 있으면) ---
        for i, c1 in enumerate(coins):
            for c2 in coins[i+1:]:
                if c1.coin_type == c2.coin_type:
                    dist = math.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)
                    touch = c1.radius + c2.radius
                    if dist < touch * 1.2:
                        # 거의 합성 직전 → 상위 레벨 점수의 절반 보너스
                        next_ct = c1.coin_type.get_next_level()
                        bonus = next_ct.score * 0.5 if next_ct else c1.coin_type.score
                        value += bonus
                    elif dist < touch * 2.5:
                        # 가까이 있음 → 작은 보너스
                        value += c1.coin_type.score * 0.1

        # --- 높은 레벨 코인 보너스 (상위 합성 가능성) ---
        for c in coins:
            value += c.coin_type.score * 0.02

        return value

    def search(
        self,
        current_coins: List[Coin],
        current_coin: CoinType,
        next_coin: Optional[CoinType] = None,
        progress_callback=None,
    ) -> Tuple[float, float, dict]:
        """
        Parallel MCTS search — uses all CPU cores for fast convergence.

        Strategy:
          Phase 1: Expand all root positions in parallel (one batch).
          Phase 2: Run parallel rollouts from each root child, batching
                   all rollout simulations across CPU cores.
          Phase 3: Push intermediate rankings to overlay for real-time
                   convergence visualization (many lines → one green line).

        Returns:
            (best_x, expected_value, details_dict)
        """
        start_time = time.time()
        positions = self._candidate_positions(current_coin)
        N = len(positions)

        # 빈 보드 특수 처리: 동전이 없으면 모든 위치가 동등 → 중앙 선호
        if not current_coins or len(current_coins) == 0:
            center = (self.wall_left + self.wall_right) / 2
            candidates = []
            for rank, x in enumerate(sorted(positions, key=lambda p: abs(p - center))[:5]):
                conf = max(0.3, 1.0 - abs(x - center) / ((self.wall_right - self.wall_left) / 2) * 0.7)
                candidates.append({
                    'x': round(x, 1), 'score': 0.0,
                    'confidence': round(conf, 3), 'rank': rank, 'visits': 1,
                })
            if progress_callback:
                progress_callback(1.0, "Empty board — center preferred")
            best = candidates[0] if candidates else {'x': center, 'score': 0.0}
            return best['x'], 0.0, {
                'iterations': 0, 'time_s': 0.0, 'depth_reached': 0,
                'children': N, 'candidates': candidates, 'batch_rounds': 0,
            }

        if progress_callback:
            progress_callback(0.05, f"Simulating {N} positions...")

        # --- Phase 1: Expand all root children in parallel ---
        try:
            finals, scores, heights = fast_sim.batch_simulate(
                current_coins, current_coin, positions,
                self.wall_left, self.wall_right, self.floor_y, self.ceiling_y,
                fast=False,
            )
        except Exception as e:
            logger.warning(f"MCTS expansion failed: {e}")
            center = (self.wall_left + self.wall_right) / 2
            return center, 0.0, {'candidates': [], 'iterations': 0}

        # Per-position accumulators
        total_values = np.zeros(N, dtype=np.float64)
        visit_counts = np.zeros(N, dtype=np.int32)
        merge_scores_base = np.array([float(scores[i]) for i in range(N)])
        child_states = finals  # List[List[Coin]]
        child_coins = [next_coin if next_coin else self._random_next_coin() for _ in range(N)]

        # Initial evaluation (no rollout yet)
        for i in range(N):
            val = self._evaluate_state(child_states[i], merge_scores_base[i])
            total_values[i] += val
            visit_counts[i] += 1

        elapsed = time.time() - start_time
        if progress_callback:
            progress_callback(0.12, f"Phase 1 done ({elapsed:.1f}s), deep rollouts...")
            # Push initial candidates for real-time display
            self._push_intermediate(positions, total_values, visit_counts,
                                    progress_callback, 0.12)

        # --- Phase 2: GPU 배치 롤아웃 ---
        total_iters = 0
        batch_round = 0

        while total_iters < self.max_iterations:
            elapsed = time.time() - start_time
            if elapsed > self.time_budget_s:
                break

            batch_round += 1

            # 각 루트 자식마다 1개 롤아웃 태스크 (위치 3개)
            rollout_tasks = []
            task_parent_idx = []

            for i in range(N):
                coins_i = child_states[i]
                if not coins_i:
                    continue
                drop_coin = self._random_next_coin()
                rpos = self._candidate_positions(drop_coin)
                if len(rpos) > 3:
                    rpos = random.sample(rpos, 3)
                rollout_tasks.append(
                    (coins_i, drop_coin, rpos,
                     self.wall_left, self.wall_right, self.floor_y, self.ceiling_y)
                )
                task_parent_idx.append(i)

            if not rollout_tasks:
                break

            # GPU 배치로 모든 롤아웃 동시 실행
            try:
                batch_results = fast_sim.batch_simulate_many(rollout_tasks, fast=True)
            except Exception as e:
                logger.warning(f"Rollout batch failed: {e}")
                break

            # 결과 처리 및 누적
            for task_idx, (r_finals, r_scores, r_heights) in enumerate(batch_results):
                parent_i = task_parent_idx[task_idx]
                if len(r_scores) == 0:
                    continue

                best_j = int(r_scores.argmax())
                step_coins = r_finals[best_j]
                step_merge = merge_scores_base[parent_i] + float(r_scores[best_j])
                val = self._evaluate_state(step_coins, step_merge)

                total_values[parent_i] += val
                visit_counts[parent_i] += 1
                total_iters += 1

            # 실시간 수렴 표시
            if progress_callback:
                pct = min(0.95, 0.12 + 0.83 * min(1.0, elapsed / self.time_budget_s))
                self._push_intermediate(positions, total_values, visit_counts,
                                        progress_callback, pct)

        # --- Phase 3: Final results ---
        elapsed = time.time() - start_time
        avg_values = total_values / np.maximum(visit_counts, 1)
        sorted_idx = np.argsort(-avg_values)

        max_val = avg_values[sorted_idx[0]] if N > 0 else 1.0
        min_val = avg_values[sorted_idx[-1]] if N > 0 else 0.0
        val_range = max(max_val - min_val, 1e-6)

        candidates = []
        for rank, idx in enumerate(sorted_idx[:5]):
            confidence = (avg_values[idx] - min_val) / val_range
            confidence = max(0.1, min(1.0, confidence))
            candidates.append({
                'x': round(positions[idx], 1),
                'score': round(float(avg_values[idx]), 1),
                'confidence': round(confidence, 3),
                'rank': rank,
                'visits': int(visit_counts[idx]),
            })

        best = candidates[0] if candidates else {'x': (self.wall_left + self.wall_right) / 2, 'score': 0}

        details = {
            'iterations': total_iters,
            'time_s': round(elapsed, 2),
            'depth_reached': 1,
            'children': N,
            'candidates': candidates,
            'batch_rounds': batch_round,
        }

        logger.info(f"MCTS: best_x={best['x']}, value={best['score']}, "
                    f"iters={total_iters}, rounds={batch_round}, time={elapsed:.2f}s, "
                    f"top={len(candidates)} candidates")

        if progress_callback:
            progress_callback(1.0, f"Done: {total_iters} iters, {elapsed:.1f}s")

        return best['x'], best['score'], details

    def _push_intermediate(self, positions, total_values, visit_counts,
                           progress_callback, pct):
        """Push intermediate candidate rankings to overlay for real-time convergence."""
        avg = total_values / np.maximum(visit_counts, 1)
        sorted_idx = np.argsort(-avg)
        max_v = avg[sorted_idx[0]]
        min_v = avg[sorted_idx[-1]]
        rng = max(max_v - min_v, 1e-6)

        top_n = min(5, len(positions))
        parts = []
        for rank in range(top_n):
            idx = sorted_idx[rank]
            conf = max(0.1, (avg[idx] - min_v) / rng)
            parts.append(f"x={positions[idx]:.0f}({conf*100:.0f}%)")

        status = f"MCTS: {' | '.join(parts)}"
        progress_callback(pct, status)

