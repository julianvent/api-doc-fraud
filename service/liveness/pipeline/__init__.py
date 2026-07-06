"""Pipeline — orchestration + pure logic.

`decision` and `scoring` are pure: they take dataclasses in, return
dataclasses out, and never touch disk, network, or models.
`orchestrator` is the only file here with side effects.
"""

from service.liveness.pipeline.decision import decide
from service.liveness.pipeline.scoring import aggregate_score

__all__ = ["aggregate_score", "decide"]
