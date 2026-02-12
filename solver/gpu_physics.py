"""
GPU-accelerated batch physics simulation using PyTorch.

Simulates hundreds of coin drop scenarios in parallel on the GPU.
Uses a simplified 2D rigid-body circle physics model:
  - Gravity, damping
  - Circle-circle collision with restitution (elasticity) and friction
  - Circle-wall collision
  - Merge detection (same-type coins touching)
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass

from models.coin import Coin, CoinType

logger = logging.getLogger(__name__)

# Coin type constants (level -> radius, score, mass)
_COIN_RADII = {ct.level: ct.radius for ct in CoinType}
_COIN_SCORES = {ct.level: ct.score for ct in CoinType}
MAX_COINS = 25  # max coins per simulation (N×N pairwise = 625)
MAX_LEVEL = 11


@dataclass
class PhysicsParams:
    """Differentiable physics parameters for auto-calibration."""
    gravity: float = 900.0       # pixels/s^2 downward (positive = down in screen coords)
    damping: float = 0.98
    elasticity: float = 0.3      # coefficient of restitution
    friction: float = 0.5
    wall_elasticity: float = 0.2
    wall_friction: float = 0.6
    dt: float = 1.0 / 60.0
    steps: int = 90              # 1.5 seconds at 60fps (enough for coins to settle)


class GPUPhysicsBatch:
    """
    Batch physics simulation on GPU.

    State tensor layout per simulation:
      coins: [MAX_COINS, 6]  ->  x, y, vx, vy, radius, level
      alive: [MAX_COINS]     ->  1.0 if coin exists, 0.0 otherwise
    """

    def __init__(self, params: PhysicsParams = None, device: str = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.params = params or PhysicsParams()
        logger.info(f"GPUPhysicsBatch on {self.device}")

    # ------------------------------------------------------------------ #
    # State encoding
    # ------------------------------------------------------------------ #
    def encode_state(self, coins: List[Coin]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a list of Coin objects into tensors. Returns (state, alive)."""
        state = torch.zeros(MAX_COINS, 6, device=self.device)
        alive = torch.zeros(MAX_COINS, device=self.device)
        for i, c in enumerate(coins[:MAX_COINS]):
            state[i] = torch.tensor([c.x, c.y, c.velocity_x, c.velocity_y,
                                     float(c.radius), float(c.coin_type.level)],
                                    device=self.device)
            alive[i] = 1.0
        return state, alive

    def decode_state(self, state: torch.Tensor, alive: torch.Tensor) -> List[Coin]:
        """Decode tensors back to Coin list."""
        coins = []
        for i in range(MAX_COINS):
            if alive[i].item() < 0.5:
                continue
            x, y, vx, vy, r, lv = state[i].tolist()
            ct = CoinType.from_level(int(round(lv)))
            if ct is None:
                continue
            coins.append(Coin(coin_type=ct, x=x, y=y, velocity_x=vx, velocity_y=vy))
        return coins

    # ------------------------------------------------------------------ #
    # Batch simulation
    # ------------------------------------------------------------------ #
    def simulate_batch(
        self,
        base_coins: List[Coin],
        drop_coin_type: CoinType,
        drop_xs: List[float],
        wall_left: float = 0.0,
        wall_right: float = 533.0,
        floor_y: float = 900.0,
        ceiling_y: float = 200.0,
    ) -> Tuple[List[List[Coin]], torch.Tensor, torch.Tensor]:
        """
        Simulate dropping a coin at each x in drop_xs, all in parallel.

        Returns:
            final_states: list of Coin lists per scenario
            scores: tensor [B] of merge scores
            max_heights: tensor [B] of highest coin y (lower = more dangerous)
        """
        B = len(drop_xs)
        p = self.params

        # Build batch state: [B, MAX_COINS, 6]
        base_state, base_alive = self.encode_state(base_coins)
        states = base_state.unsqueeze(0).expand(B, -1, -1).clone()
        alives = base_alive.unsqueeze(0).expand(B, -1).clone()

        # Find first free slot for the dropped coin
        n_base = int(base_alive.sum().item())
        drop_idx = min(n_base, MAX_COINS - 1)

        drop_xs_t = torch.tensor(drop_xs, device=self.device, dtype=torch.float32)
        drop_y = ceiling_y - 50.0  # drop from above ceiling
        drop_r = float(drop_coin_type.radius)
        drop_lv = float(drop_coin_type.level)

        states[:, drop_idx, 0] = drop_xs_t
        states[:, drop_idx, 1] = drop_y
        states[:, drop_idx, 2] = 0.0
        states[:, drop_idx, 3] = 0.0
        states[:, drop_idx, 4] = drop_r
        states[:, drop_idx, 5] = drop_lv
        alives[:, drop_idx] = 1.0

        merge_scores = torch.zeros(B, device=self.device)

        # Run physics steps
        for step in range(p.steps):
            states, alives, step_scores = self._step(
                states, alives, wall_left, wall_right, floor_y, p
            )
            merge_scores += step_scores

        # Decode results
        final_states = []
        max_heights = torch.full((B,), floor_y, device=self.device)

        for b in range(B):
            coins_out = self.decode_state(states[b], alives[b])
            final_states.append(coins_out)
            # Highest coin = smallest y (screen coords, y-down)
            alive_mask = alives[b] > 0.5
            if alive_mask.any():
                ys = states[b, :, 1][alive_mask] - states[b, :, 4][alive_mask]
                max_heights[b] = ys.min()

        return final_states, merge_scores, max_heights

    @torch.no_grad()
    def _step(
        self,
        states: torch.Tensor,   # [B, N, 6]
        alives: torch.Tensor,   # [B, N]
        wall_left: float,
        wall_right: float,
        floor_y: float,
        p: PhysicsParams,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One fully-vectorized physics step for the whole batch."""
        B, N, _ = states.shape
        dt = p.dt

        x  = states[:, :, 0].clone()
        y  = states[:, :, 1].clone()
        vx = states[:, :, 2].clone()
        vy = states[:, :, 3].clone()
        r  = states[:, :, 4]
        lv = states[:, :, 5].clone()
        alive_f = alives.clone()   # float mask

        # --- Gravity + Damping + Position update ---
        vy = (vy + p.gravity * dt) * p.damping
        vx = vx * p.damping
        x = x + vx * dt
        y = y + vy * dt

        alive_mask = alive_f > 0.5

        # --- Wall collisions (fully vectorized) ---
        hit_left = ((wall_left + r - x) > 0) & alive_mask
        x = torch.where(hit_left, wall_left + r, x)
        vx = torch.where(hit_left, vx.abs() * p.wall_elasticity, vx)

        hit_right = ((x + r - wall_right) > 0) & alive_mask
        x = torch.where(hit_right, wall_right - r, x)
        vx = torch.where(hit_right, -vx.abs() * p.wall_elasticity, vx)

        hit_floor = ((y + r - floor_y) > 0) & alive_mask
        y = torch.where(hit_floor, floor_y - r, y)
        vy = torch.where(hit_floor, -vy.abs() * p.elasticity, vy)
        vx = torch.where(hit_floor, vx * (1.0 - p.friction * 0.1), vx)

        # --- Pairwise collision detection (vectorized) ---
        dx = x.unsqueeze(2) - x.unsqueeze(1)   # [B, N, N]
        dy = y.unsqueeze(2) - y.unsqueeze(1)
        dist_sq = dx**2 + dy**2
        dist = torch.sqrt(dist_sq + 1e-8)
        sum_r = r.unsqueeze(2) + r.unsqueeze(1)
        overlap = (sum_r - dist).clamp(min=0)

        # Upper-triangle, both alive
        tri = torch.triu(torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1)
        alive_pair = (alive_f.unsqueeze(2) > 0.5) & (alive_f.unsqueeze(1) > 0.5)
        active = (overlap > 0) & tri.unsqueeze(0) & alive_pair

        same_lv = (lv.unsqueeze(2) - lv.unsqueeze(1)).abs() < 0.5

        # --- Vectorized non-merge collision response ---
        non_merge = active & (~same_lv)
        # Normal vector
        nx = dx / dist.clamp(min=1e-4)
        ny = dy / dist.clamp(min=1e-4)
        # Mass ~ r^2
        mi = r.unsqueeze(2) ** 2   # [B, N, 1] broadcast to [B, N, N]
        mj = r.unsqueeze(1) ** 2
        total_m = mi + mj + 1e-8

        # Position correction (push apart)
        push_x = nx * overlap * (mj / total_m) * 0.5 * non_merge.float()
        push_y = ny * overlap * (mj / total_m) * 0.5 * non_merge.float()
        x += push_x.sum(dim=2) - push_x.sum(dim=1)
        y += push_y.sum(dim=2) - push_y.sum(dim=1)

        # Velocity impulse
        dvx = vx.unsqueeze(2) - vx.unsqueeze(1)
        dvy = vy.unsqueeze(2) - vy.unsqueeze(1)
        rel_vn = dvx * nx + dvy * ny
        # Only apply when approaching (rel_vn < 0)
        impulse_mask = non_merge & (rel_vn < 0)
        inv_m = 1.0 / mi + 1.0 / mj
        j_imp = -(1 + p.elasticity) * rel_vn / inv_m.clamp(min=1e-8)
        j_imp = j_imp * impulse_mask.float()
        imp_x = j_imp / mi.clamp(min=1e-4) * nx
        imp_y = j_imp / mi.clamp(min=1e-4) * ny
        vx += imp_x.sum(dim=2) - imp_x.sum(dim=1)
        vy += imp_y.sum(dim=2) - imp_y.sum(dim=1)

        # --- Process merges (greedy: one merge per coin per step) ---
        merge_scores = torch.zeros(B, device=x.device)
        mergeable = active & same_lv

        if mergeable.any():
            # Score lookup tensor
            score_lut = torch.zeros(MAX_LEVEL + 2, device=x.device)
            for lvl, sc in _COIN_SCORES.items():
                score_lut[lvl] = sc
            radius_lut = torch.zeros(MAX_LEVEL + 2, device=x.device)
            for lvl, rd in _COIN_RADII.items():
                radius_lut[lvl] = rd

            # Process per batch (unavoidable for greedy merge, but inner loop is minimal)
            for b in range(B):
                pairs = mergeable[b].nonzero(as_tuple=False)
                if pairs.shape[0] == 0:
                    continue
                used = set()
                for k in range(min(pairs.shape[0], 20)):  # cap merges per step
                    i, j = pairs[k, 0].item(), pairs[k, 1].item()
                    if i in used or j in used or alive_f[b, i] < 0.5 or alive_f[b, j] < 0.5:
                        continue
                    cl = int(round(lv[b, i].item()))
                    nl = cl + 1
                    if nl <= MAX_LEVEL:
                        # 합성 후 생성되는 상위 레벨 코인의 점수
                        merge_scores[b] += score_lut[nl]
                        x[b, i] = (x[b, i] + x[b, j]) * 0.5
                        y[b, i] = (y[b, i] + y[b, j]) * 0.5
                        vx[b, i] = 0; vy[b, i] = 0
                        r[b, i] = radius_lut[nl]
                        lv[b, i] = float(nl)
                    else:
                        alive_f[b, i] = 0
                        merge_scores[b] += score_lut[cl] * 2
                    alive_f[b, j] = 0
                    used.add(i); used.add(j)

        # Write back
        states[:, :, 0] = x
        states[:, :, 1] = y
        states[:, :, 2] = vx
        states[:, :, 3] = vy
        states[:, :, 4] = r
        states[:, :, 5] = lv
        alives.copy_(alive_f)

        return states, alives, merge_scores


    def simulate_batch_fast(
        self,
        base_coins: List[Coin],
        drop_coin_type: CoinType,
        drop_xs: List[float],
        wall_left: float = 0.0,
        wall_right: float = 533.0,
        floor_y: float = 900.0,
        ceiling_y: float = 200.0,
    ) -> Tuple[List[List[Coin]], torch.Tensor, torch.Tensor]:
        """
        Fast simulation for MCTS rollouts — fewer steps, larger dt.
        Returns same format as simulate_batch.
        """
        B = len(drop_xs)
        p = self.params

        base_state, base_alive = self.encode_state(base_coins)
        states = base_state.unsqueeze(0).expand(B, -1, -1).clone()
        alives = base_alive.unsqueeze(0).expand(B, -1).clone()

        n_base = int(base_alive.sum().item())
        drop_idx = min(n_base, MAX_COINS - 1)

        drop_xs_t = torch.tensor(drop_xs, device=self.device, dtype=torch.float32)
        states[:, drop_idx, 0] = drop_xs_t
        states[:, drop_idx, 1] = ceiling_y - 50.0
        states[:, drop_idx, 2] = 0.0
        states[:, drop_idx, 3] = 0.0
        states[:, drop_idx, 4] = float(drop_coin_type.radius)
        states[:, drop_idx, 5] = float(drop_coin_type.level)
        alives[:, drop_idx] = 1.0

        merge_scores = torch.zeros(B, device=self.device)

        # Fast: 30 steps with 3x dt
        fast_params = PhysicsParams(
            gravity=p.gravity, damping=p.damping, elasticity=p.elasticity,
            friction=p.friction, wall_elasticity=p.wall_elasticity,
            wall_friction=p.wall_friction, dt=p.dt * 3.0, steps=30
        )

        for _ in range(fast_params.steps):
            states, alives, step_scores = self._step(
                states, alives, wall_left, wall_right, floor_y, fast_params
            )
            merge_scores += step_scores

        final_states = []
        max_heights = torch.full((B,), floor_y, device=self.device)
        for b in range(B):
            coins_out = self.decode_state(states[b], alives[b])
            final_states.append(coins_out)
            alive_mask = alives[b] > 0.5
            if alive_mask.any():
                ys = states[b, :, 1][alive_mask] - states[b, :, 4][alive_mask]
                max_heights[b] = ys.min()

        return final_states, merge_scores, max_heights

    @torch.no_grad()
    def simulate_many_fast(
        self,
        tasks: list,
        wall_left: float = 0.0,
        wall_right: float = 533.0,
        floor_y: float = 900.0,
    ) -> list:
        """
        여러 태스크를 하나의 GPU 배치로 합쳐서 실행.
        각 task = (base_coins, drop_coin_type, drop_xs, ceiling_y)
        Returns: list of (final_states, merge_scores_tensor, max_heights_tensor)

        GPU 커널 호출 1번으로 모든 시뮬레이션 동시 실행 → 오버헤드 최소화.
        """
        p = self.params
        fast_params = PhysicsParams(
            gravity=p.gravity, damping=p.damping, elasticity=p.elasticity,
            friction=p.friction, wall_elasticity=p.wall_elasticity,
            wall_friction=p.wall_friction, dt=p.dt * 3.0, steps=30
        )

        # 모든 태스크의 시뮬레이션을 하나의 배치 차원으로 합침
        all_states = []
        all_alives = []
        task_ranges = []  # (start, count) per task
        offset = 0

        for base_coins, drop_coin_type, drop_xs, ceiling_y in tasks:
            base_state, base_alive = self.encode_state(base_coins)
            B = len(drop_xs)
            states = base_state.unsqueeze(0).expand(B, -1, -1).clone()
            alives = base_alive.unsqueeze(0).expand(B, -1).clone()

            n_base = int(base_alive.sum().item())
            drop_idx = min(n_base, MAX_COINS - 1)

            drop_xs_t = torch.tensor(drop_xs, device=self.device, dtype=torch.float32)
            states[:, drop_idx, 0] = drop_xs_t
            states[:, drop_idx, 1] = ceiling_y - 50.0
            states[:, drop_idx, 2] = 0.0
            states[:, drop_idx, 3] = 0.0
            states[:, drop_idx, 4] = float(drop_coin_type.radius)
            states[:, drop_idx, 5] = float(drop_coin_type.level)
            alives[:, drop_idx] = 1.0

            all_states.append(states)
            all_alives.append(alives)
            task_ranges.append((offset, B))
            offset += B

        if offset == 0:
            return []

        # 하나의 거대 배치로 합침
        big_states = torch.cat(all_states, dim=0)   # [total_B, N, 6]
        big_alives = torch.cat(all_alives, dim=0)   # [total_B, N]
        merge_scores = torch.zeros(offset, device=self.device)

        # GPU 물리 스텝 실행 (1번의 배치로 전부)
        for _ in range(fast_params.steps):
            big_states, big_alives, step_scores = self._step(
                big_states, big_alives, wall_left, wall_right, floor_y, fast_params
            )
            merge_scores += step_scores

        # 태스크별로 결과 분리
        output = []
        for start, count in task_ranges:
            final_states = []
            max_heights = torch.full((count,), floor_y, device=self.device)
            for b in range(count):
                idx = start + b
                coins_out = self.decode_state(big_states[idx], big_alives[idx])
                final_states.append(coins_out)
                alive_mask = big_alives[idx] > 0.5
                if alive_mask.any():
                    ys = big_states[idx, :, 1][alive_mask] - big_states[idx, :, 4][alive_mask]
                    max_heights[b] = ys.min()
            output.append((final_states, merge_scores[start:start+count], max_heights))

        return output


class PhysicsCalibrator:
    """
    Auto-calibrate physics parameters by comparing simulation predictions
    with actual game outcomes. Uses gradient-free optimization (Nelder-Mead)
    to minimize prediction error.
    """

    def __init__(self, gpu_physics: GPUPhysicsBatch):
        self.gpu = gpu_physics
        self.history: List[dict] = []  # {before, drop, predicted, actual, params}

    def record(self, before: List[Coin], drop_coin: CoinType, drop_x: float,
               actual_after: List[Coin]):
        """Record a real game outcome for calibration."""
        self.history.append({
            'before': before,
            'drop_coin': drop_coin,
            'drop_x': drop_x,
            'actual': actual_after,
        })
        if len(self.history) > 50:
            self.history = self.history[-50:]

    def calibrate(self, wall_left: float, wall_right: float, floor_y: float) -> PhysicsParams:
        """Run calibration on recent history. Returns optimized params."""
        if len(self.history) < 3:
            return self.gpu.params

        from scipy.optimize import minimize

        base_params = self.gpu.params

        def objective(x):
            p = PhysicsParams(
                gravity=x[0], damping=x[1], elasticity=x[2],
                friction=x[3], wall_elasticity=x[4], wall_friction=x[5],
                dt=base_params.dt, steps=base_params.steps
            )
            self.gpu.params = p
            total_error = 0.0
            for rec in self.history[-10:]:
                finals, _, _ = self.gpu.simulate_batch(
                    rec['before'], rec['drop_coin'], [rec['drop_x']],
                    wall_left, wall_right, floor_y
                )
                pred = finals[0]
                actual = rec['actual']
                total_error += self._state_error(pred, actual)
            return total_error / len(self.history[-10:])

        x0 = [base_params.gravity, base_params.damping, base_params.elasticity,
              base_params.friction, base_params.wall_elasticity, base_params.wall_friction]
        bounds = [(400, 1400), (0.8, 0.999), (0.05, 0.8),
                  (0.1, 0.9), (0.05, 0.6), (0.1, 0.9)]

        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                            options={'maxiter': 50, 'xatol': 0.01})
            best = result.x
            optimized = PhysicsParams(
                gravity=best[0], damping=best[1], elasticity=best[2],
                friction=best[3], wall_elasticity=best[4], wall_friction=best[5],
                dt=base_params.dt, steps=base_params.steps
            )
            self.gpu.params = optimized
            logger.info(f"Calibration done: gravity={best[0]:.1f}, elast={best[2]:.3f}, "
                       f"fric={best[3]:.3f}, error={result.fun:.2f}")
            return optimized
        except Exception as e:
            logger.warning(f"Calibration failed: {e}")
            self.gpu.params = base_params
            return base_params

    def save(self, path: str = 'calibration.json'):
        """캘리브레이션 결과를 파일에 저장"""
        import json
        try:
            p = self.gpu.params
            data = {
                'gravity': p.gravity, 'damping': p.damping,
                'elasticity': p.elasticity, 'friction': p.friction,
                'wall_elasticity': p.wall_elasticity, 'wall_friction': p.wall_friction,
                'history_count': len(self.history),
            }
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Calibration saved: {path}")
        except Exception as e:
            logger.warning(f"Calibration save failed: {e}")

    def load(self, path: str = 'calibration.json') -> bool:
        """저장된 캘리브레이션 결과 로드"""
        import json
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            self.gpu.params = PhysicsParams(
                gravity=data['gravity'], damping=data['damping'],
                elasticity=data['elasticity'], friction=data['friction'],
                wall_elasticity=data['wall_elasticity'],
                wall_friction=data['wall_friction'],
                dt=self.gpu.params.dt, steps=self.gpu.params.steps,
            )
            logger.info(f"Calibration loaded: gravity={data['gravity']:.1f}, "
                       f"elast={data['elasticity']:.3f}, fric={data['friction']:.3f}")
            return True
        except FileNotFoundError:
            return False
        except Exception as e:
            logger.warning(f"Calibration load failed: {e}")
            return False

    def _state_error(self, predicted: List[Coin], actual: List[Coin]) -> float:
        """Compute error between predicted and actual states."""
        if not actual:
            return 0.0 if not predicted else 100.0

        # Match coins by nearest distance
        error = 0.0
        used = set()
        for ac in actual:
            best_dist = float('inf')
            best_idx = -1
            for i, pr in enumerate(predicted):
                if i in used:
                    continue
                if pr.coin_type.level != ac.coin_type.level:
                    continue
                d = math.sqrt((pr.x - ac.x)**2 + (pr.y - ac.y)**2)
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            if best_idx >= 0:
                used.add(best_idx)
                error += best_dist
            else:
                error += 200.0  # unmatched coin penalty

        # Penalty for extra predicted coins
        error += max(0, len(predicted) - len(actual)) * 100.0
        return error
