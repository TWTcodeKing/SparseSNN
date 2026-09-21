"""Schedule builder: adapts IR to BA-MTTS scheduler.

Self-contained — no imports from sengine/.
"""

from sengine_cpu.ir import EngineIR
from sengine_cpu.scheduler import build_schedule_from_ir


def build_schedule(ir: EngineIR) -> list[int]:
    """Build BA-MTTS execution schedule from optimized IR."""
    return build_schedule_from_ir(ir)
