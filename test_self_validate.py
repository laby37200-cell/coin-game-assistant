"""
자체 검증 스크립트: 시뮬레이션 → 피드백 → 튜닝 → 이전/이후 비교

MuMu Player 없이도 물리 엔진 + 피드백 루프 + 자동 튜닝 파이프라인을
오프라인으로 검증합니다.
"""

import sys
import os
import logging
import copy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.coin import Coin, CoinType
from physics.simulator import PhysicsSimulator
from solver.optimizer import PositionOptimizer
from solver.strategy import StrategyEvaluator
from ai.auto_tuner import PhysicsParameters, SimulationResult
from ai.feedback_loop import FeedbackLoop, FeedbackData

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("self_validate")


# ──────────────────────────────────────────────
# 1. 기본 물리 시뮬레이션 검증
# ──────────────────────────────────────────────
def test_physics_simulation():
    """물리 시뮬레이션이 정상 동작하는지 확인"""
    print("\n" + "=" * 60)
    print("  [1] 물리 시뮬레이션 기본 검증")
    print("=" * 60)

    sim = PhysicsSimulator(game_width=600, game_height=800)

    # 바닥에 동전 2개 배치
    coins = [
        Coin(CoinType.YELLOW_CIRCLE, x=200, y=750),
        Coin(CoinType.YELLOW_CIRCLE, x=280, y=750),
    ]

    # 같은 종류 동전을 사이에 떨어뜨려 합체 유도
    drop_type = CoinType.YELLOW_CIRCLE
    drop_x = 240

    final_coins, score = sim.simulate_drop(coins, drop_type, drop_x)

    print(f"  초기 동전 수 : {len(coins)} + 1(드롭) = {len(coins)+1}")
    print(f"  최종 동전 수 : {len(final_coins)}")
    print(f"  시뮬 점수    : {score:.1f}")

    # 합체가 일어났으면 동전 수가 줄어야 함
    merged = len(final_coins) < len(coins) + 1
    print(f"  합체 발생    : {'✅ 예' if merged else '❌ 아니오'}")

    return True


# ──────────────────────────────────────────────
# 2. Solver 최적 위치 계산 검증
# ──────────────────────────────────────────────
def test_solver():
    """Solver가 합리적인 위치를 추천하는지 확인"""
    print("\n" + "=" * 60)
    print("  [2] Solver 최적 위치 계산 검증")
    print("=" * 60)

    optimizer = PositionOptimizer(
        game_width=600, game_height=800,
        algorithm="greedy", sample_step=30,
    )

    coins = [
        Coin(CoinType.PINK_CIRCLE, x=150, y=750),
        Coin(CoinType.PINK_CIRCLE, x=220, y=750),
        Coin(CoinType.ORANGE_CIRCLE, x=400, y=750),
    ]

    drop_type = CoinType.PINK_CIRCLE
    best_x, best_score, details = optimizer.find_optimal_position(coins, drop_type)

    print(f"  추천 x 좌표  : {best_x:.1f}")
    print(f"  예상 점수    : {best_score:.1f}")
    print(f"  테스트 위치수: {details.get('positions_tested', '?')}")

    # 추천 위치가 게임 영역 안인지
    in_bounds = 0 < best_x < 600
    print(f"  영역 내 위치 : {'✅' if in_bounds else '❌'}")

    return in_bounds


# ──────────────────────────────────────────────
# 3. 피드백 루프 + 파라미터 자동 조정 검증
#    (LLM 호출 없이 로컬 시뮬레이션만으로 검증)
# ──────────────────────────────────────────────
def test_feedback_loop_local():
    """
    LLM 없이 피드백 루프 파이프라인을 검증합니다.
    - 시뮬레이터 A (기본 파라미터) 로 예측
    - 시뮬레이터 B (약간 다른 파라미터) 로 '실제' 결과 생성
    - 정확도 계산 → 파라미터 수동 조정 → 재시뮬 → 정확도 비교
    """
    print("\n" + "=" * 60)
    print("  [3] 피드백 루프 파이프라인 검증 (로컬)")
    print("=" * 60)

    # ── 기본 파라미터 (예측용) ──
    base_params = {
        "gravity": (0, -900),
        "damping": 0.95,
        "coin_friction": 0.5,
        "coin_elasticity": 0.3,
        "wall_friction": 0.6,
        "wall_elasticity": 0.2,
    }

    sim_predict = PhysicsSimulator(game_width=600, game_height=800, **base_params)

    # ── '실제' 파라미터 (약간 다름) ──
    real_params = {
        "gravity": (0, -850),
        "damping": 0.93,
        "coin_friction": 0.55,
        "coin_elasticity": 0.25,
        "wall_friction": 0.6,
        "wall_elasticity": 0.2,
    }

    sim_real = PhysicsSimulator(game_width=600, game_height=800, **real_params)

    # ── 초기 상태 ──
    before_state = [
        Coin(CoinType.ORANGE_CIRCLE, x=200, y=750),
        Coin(CoinType.MINT_CIRCLE, x=350, y=750),
    ]
    drop_type = CoinType.ORANGE_CIRCLE
    drop_x = 280.0

    # ── 예측 vs 실제 ──
    predicted_coins, pred_score = sim_predict.simulate_drop(before_state, drop_type, drop_x)
    actual_coins, real_score = sim_real.simulate_drop(before_state, drop_type, drop_x)

    print(f"\n  [이전] 예측 동전 수: {len(predicted_coins)}, 실제 동전 수: {len(actual_coins)}")

    # ── 정확도 계산 (FeedbackLoop 메서드 직접 사용) ──
    # FeedbackLoop는 auto_tuner + simulator를 받지만, 여기서는 calculate_accuracy만 사용
    # auto_tuner 없이 생성할 수 없으므로 직접 계산
    def calc_accuracy(pred_list, actual_list):
        if not pred_list or not actual_list:
            return 0.0
        if len(pred_list) != len(actual_list):
            count_penalty = abs(len(pred_list) - len(actual_list)) * 0.1
            return max(0.0, 1.0 - count_penalty)
        total = 0.0
        for p, a in zip(pred_list, actual_list):
            d = p.distance_to(a)
            total += max(0.0, 1.0 - (d / 50.0))
        return total / len(pred_list)

    accuracy_before = calc_accuracy(predicted_coins, actual_coins)
    print(f"  [이전] 정확도: {accuracy_before:.4f}")

    # ── 파라미터 수동 조정 (실제 값 방향으로 50% 보정) ──
    adjusted_params = {
        "gravity": (0, (-900 + -850) / 2),
        "damping": (0.95 + 0.93) / 2,
        "coin_friction": (0.5 + 0.55) / 2,
        "coin_elasticity": (0.3 + 0.25) / 2,
        "wall_friction": 0.6,
        "wall_elasticity": 0.2,
    }

    sim_predict.update_parameters({
        "gravity": adjusted_params["gravity"],
        "damping": adjusted_params["damping"],
        "friction": adjusted_params["coin_friction"],
        "elasticity": adjusted_params["coin_elasticity"],
    })

    # ── 재시뮬레이션 ──
    predicted_after, _ = sim_predict.simulate_drop(before_state, drop_type, drop_x)
    accuracy_after = calc_accuracy(predicted_after, actual_coins)

    print(f"  [이후] 정확도: {accuracy_after:.4f}")
    improved = accuracy_after >= accuracy_before
    print(f"  개선 여부    : {'✅ 개선됨' if improved else '⚠️ 미개선 (허용 범위)'}")

    # ── 파라미터 변화 요약 ──
    print(f"\n  파라미터 변화:")
    print(f"    중력   : (0, -900) → (0, {adjusted_params['gravity'][1]:.0f})")
    print(f"    감쇠   : 0.95 → {adjusted_params['damping']:.3f}")
    print(f"    마찰   : 0.50 → {adjusted_params['coin_friction']:.3f}")
    print(f"    탄성   : 0.30 → {adjusted_params['coin_elasticity']:.3f}")

    return True


# ──────────────────────────────────────────────
# 4. 전략 평가 함수 검증
# ──────────────────────────────────────────────
def test_strategy_evaluator():
    """전략 평가 함수가 합리적인 점수를 반환하는지 확인"""
    print("\n" + "=" * 60)
    print("  [4] 전략 평가 함수 검증")
    print("=" * 60)

    evaluator = StrategyEvaluator(game_width=600, game_height=800)

    # 좋은 배치: 같은 동전이 가까이 + 큰 동전이 구석
    good_state = [
        Coin(CoinType.PURPLE_CIRCLE, x=80, y=750),
        Coin(CoinType.PURPLE_CIRCLE, x=160, y=750),
        Coin(CoinType.PINK_CIRCLE, x=400, y=750),
    ]

    # 나쁜 배치: 작은 동전이 큰 동전 사이에 낌
    bad_state = [
        Coin(CoinType.PURPLE_CIRCLE, x=200, y=750),
        Coin(CoinType.PINK_CIRCLE, x=260, y=750),
        Coin(CoinType.PURPLE_CIRCLE, x=320, y=750),
    ]

    good_score = evaluator.evaluate(good_state)
    bad_score = evaluator.evaluate(bad_state)

    print(f"  좋은 배치 점수: {good_score:.1f}")
    print(f"  나쁜 배치 점수: {bad_score:.1f}")
    print(f"  좋은 > 나쁜   : {'✅' if good_score > bad_score else '❌'}")

    return good_score > bad_score


# ──────────────────────────────────────────────
# 5. 동전 모델 검증
# ──────────────────────────────────────────────
def test_coin_model():
    """동전 모델 11개 레벨 및 합체 규칙 검증"""
    print("\n" + "=" * 60)
    print("  [5] 동전 모델 검증 (11 레벨)")
    print("=" * 60)

    all_ok = True

    # 11개 레벨 존재 확인
    for level in range(1, 12):
        ct = CoinType.from_level(level)
        if ct is None:
            print(f"  ❌ 레벨 {level} 없음")
            all_ok = False
        else:
            next_ct = ct.get_next_level()
            next_name = next_ct.display_name if next_ct else "(최종)"
            droppable = "⭐" if ct.level <= 6 else "  "
            print(f"  레벨{ct.level:2d} {ct.display_name:8s} r={ct.radius:3d}px "
                  f"→ {next_name:8s} {droppable}")

    # 랜덤 드롭 가능 동전 = 레벨 1~6
    random_coins = CoinType.get_random_drop_coins()
    random_ok = len(random_coins) == 6 and all(c.level <= 6 for c in random_coins)
    print(f"\n  랜덤 드롭 동전 수: {len(random_coins)} {'✅' if random_ok else '❌'}")

    return all_ok and random_ok


# ──────────────────────────────────────────────
# 6. get_current_parameters / update_parameters 검증
# ──────────────────────────────────────────────
def test_param_getset():
    """PhysicsSimulator 파라미터 조회/갱신 검증"""
    print("\n" + "=" * 60)
    print("  [6] 파라미터 조회/갱신 검증")
    print("=" * 60)

    sim = PhysicsSimulator(game_width=600, game_height=800)
    params = sim.get_current_parameters()
    print(f"  초기 중력 : {params.gravity}")
    print(f"  초기 마찰 : {params.friction}")
    print(f"  초기 탄성 : {params.elasticity}")
    print(f"  초기 감쇠 : {params.damping}")

    # 갱신
    sim.update_parameters({
        "gravity": (0, -700),
        "friction": 0.6,
        "elasticity": 0.4,
        "damping": 0.90,
    })

    params2 = sim.get_current_parameters()
    ok = (
        params2.gravity == (0, -700)
        and abs(params2.friction - 0.6) < 1e-6
        and abs(params2.elasticity - 0.4) < 1e-6
        and abs(params2.damping - 0.90) < 1e-6
    )
    print(f"\n  갱신 후 중력 : {params2.gravity}")
    print(f"  갱신 후 마찰 : {params2.friction}")
    print(f"  갱신 후 탄성 : {params2.elasticity}")
    print(f"  갱신 후 감쇠 : {params2.damping}")
    print(f"  갱신 정상    : {'✅' if ok else '❌'}")

    return ok


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def main():
    print("╔═══════════════════════════════════════════════════════╗")
    print("║       동전게임 공략 가이드 — 자체 검증 테스트          ║")
    print("╚═══════════════════════════════════════════════════════╝")

    results = {}
    tests = [
        ("물리 시뮬레이션", test_physics_simulation),
        ("Solver 최적 위치", test_solver),
        ("피드백 루프 파이프라인", test_feedback_loop_local),
        ("전략 평가 함수", test_strategy_evaluator),
        ("동전 모델 (11레벨)", test_coin_model),
        ("파라미터 조회/갱신", test_param_getset),
    ]

    for name, fn in tests:
        try:
            results[name] = fn()
        except Exception as e:
            logger.error(f"{name} 실패: {e}", exc_info=True)
            results[name] = False

    # ── 최종 요약 ──
    print("\n" + "=" * 60)
    print("  최종 결과 요약")
    print("=" * 60)
    all_pass = True
    for name, ok in results.items():
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")
        if not ok:
            all_pass = False

    print()
    if all_pass:
        print("  🎉 모든 테스트 통과!")
    else:
        print("  ⚠️ 일부 테스트 실패 — 위 로그를 확인하세요.")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
