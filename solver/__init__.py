"""Solver package for optimal position calculation."""
try:
    from .strategy import StrategyEvaluator
    from .optimizer import PositionOptimizer
    __all__ = ['StrategyEvaluator', 'PositionOptimizer']
except (ImportError, ModuleNotFoundError):
    # LLM-only 빌드에서는 physics 의존 모듈이 없으므로 무시
    __all__ = []
