"""
GPU 기반 물리 시뮬레이터 — MCTS 롤아웃용.

모든 시뮬레이션을 GPU(torch)에서 배치로 실행:
  - CPU 멀티프로세스 없음 → CPU 과부하 방지
  - GPUPhysicsBatch.simulate_batch_fast 사용
  - 점수 = 합성 시 상위 레벨 코인 점수 (단순 드롭은 0점)
"""

import numpy as np
import logging
from typing import List, Tuple, Optional

from models.coin import Coin, CoinType

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #
_MAX_COINS = 25
_MAX_LEVEL = 11
_RADII = np.zeros(_MAX_LEVEL + 2, dtype=np.float32)
_SCORES = np.zeros(_MAX_LEVEL + 2, dtype=np.float32)
for _ct in CoinType:
    _RADII[_ct.level] = _ct.radius
    _SCORES[_ct.level] = _ct.score

# GPU 인스턴스 (lazy init)
_gpu = None


def _get_gpu():
    """GPUPhysicsBatch 싱글턴 (lazy init)."""
    global _gpu
    if _gpu is None:
        from solver.gpu_physics import GPUPhysicsBatch
        _gpu = GPUPhysicsBatch()
        logger.info(f"fast_sim GPU 초기화: {_gpu.device}")
    return _gpu


def warmup_pool():
    """GPU 워밍업 — 호환성을 위해 이름 유지."""
    gpu = _get_gpu()
    logger.info(f"fast_sim GPU 워밍업 완료: {gpu.device}")


# ------------------------------------------------------------------ #
# Encode / Decode
# ------------------------------------------------------------------ #
def encode_coins(coins: List[Coin]) -> Tuple[np.ndarray, np.ndarray]:
    """Encode coins into numpy arrays. Returns (state[N,5], alive[N])."""
    state = np.zeros((_MAX_COINS, 5), dtype=np.float32)
    alive = np.zeros(_MAX_COINS, dtype=np.float32)
    for i, c in enumerate(coins[:_MAX_COINS]):
        state[i] = [c.x, c.y, c.velocity_x, c.velocity_y, c.coin_type.level]
        alive[i] = 1.0
    return state, alive


def decode_coins(state: np.ndarray, alive: np.ndarray) -> List[Coin]:
    """Decode numpy arrays back to Coin list."""
    coins = []
    for i in range(_MAX_COINS):
        if alive[i] < 0.5:
            continue
        x, y, vx, vy, lv = state[i]
        ct = CoinType.from_level(int(round(lv)))
        if ct is None:
            continue
        coins.append(Coin(coin_type=ct, x=float(x), y=float(y),
                          velocity_x=float(vx), velocity_y=float(vy)))
    return coins


# ------------------------------------------------------------------ #
# Public API — GPU 배치 시뮬레이션
# ------------------------------------------------------------------ #
def batch_simulate(
    coins: List[Coin],
    drop_coin_type: CoinType,
    drop_xs: List[float],
    wall_left: float = 105.0,
    wall_right: float = 435.0,
    floor_y: float = 870.0,
    ceiling_y: float = 200.0,
    fast: bool = True,
    **kwargs,
) -> Tuple[List[List[Coin]], np.ndarray, np.ndarray]:
    """
    GPU 배치 시뮬레이션 — 모든 드롭 위치를 GPU에서 동시에 실행.

    Returns:
        (final_coin_lists, merge_scores_array, max_heights_array)
    """
    gpu = _get_gpu()

    if fast:
        finals, scores_t, heights_t = gpu.simulate_batch_fast(
            coins, drop_coin_type, drop_xs,
            wall_left, wall_right, floor_y, ceiling_y)
    else:
        finals, scores_t, heights_t = gpu.simulate_batch(
            coins, drop_coin_type, drop_xs,
            wall_left, wall_right, floor_y, ceiling_y)

    # torch tensor → numpy
    scores_arr = scores_t.cpu().numpy().astype(np.float32)
    heights_arr = heights_t.cpu().numpy().astype(np.float32)

    return finals, scores_arr, heights_arr


def batch_simulate_many(
    tasks: List[Tuple],
    fast: bool = True,
) -> List[Tuple[List[List[Coin]], np.ndarray, np.ndarray]]:
    """
    여러 독립 시뮬레이션을 하나의 GPU 배치로 합쳐서 실행.
    각 task = (coins, drop_coin_type, drop_xs, wall_left, wall_right, floor_y, ceiling_y)
    Returns: list of (final_coin_lists, scores, heights) per task.
    """
    if not tasks:
        return []

    gpu = _get_gpu()

    # 공통 벽/바닥 좌표 (첫 태스크 기준)
    _, _, _, wl0, wr0, fy0, cy0 = tasks[0]

    # simulate_many_fast 형식으로 변환
    gpu_tasks = []
    for coins, drop_coin_type, drop_xs, wl, wr, fy, cy in tasks:
        gpu_tasks.append((coins, drop_coin_type, drop_xs, cy))

    try:
        raw_results = gpu.simulate_many_fast(gpu_tasks, wl0, wr0, fy0)
    except Exception as e:
        logger.warning(f"simulate_many_fast 실패, 개별 실행으로 폴백: {e}")
        output = []
        for coins, drop_coin_type, drop_xs, wl, wr, fy, cy in tasks:
            result = batch_simulate(coins, drop_coin_type, drop_xs,
                                    wl, wr, fy, cy, fast=fast)
            output.append(result)
        return output

    # torch tensor → numpy 변환
    output = []
    for finals, scores_t, heights_t in raw_results:
        scores_arr = scores_t.cpu().numpy().astype(np.float32)
        heights_arr = heights_t.cpu().numpy().astype(np.float32)
        output.append((finals, scores_arr, heights_arr))

    return output
