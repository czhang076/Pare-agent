# Open Questions (Tier 2: Resolve during Phase 2.3 - 2.4)

These questions address specific implementation edge cases that need to be resolved while building the Phase 2.3 `recovery_detector_v2.py` and Phase 2.4 `classifier_liu.py`. They do not block the Phase 2.1 schema work.

1. **Recovery window distance limit:** What is the maximum allowed distance (in turns or `global_index`) between an error and a valid correction? We need a boundary to prevent an unrelated edit 10 turns later from being falsely flagged as a recovery.
2. **L2 "same error_signal" scope rules:** How do we link an L2 tactical switch correction to its triggering error? Does the error signal's scope persist until the next successful tool call, or does it decay after $N$ turns?
3. **Module B "toxic" definition refinement:** Should the toxic label strictly use only C2 (Premature Success), or should it exclude/include B2.2 (Syntax Error in final state)?
4. **B2.2 detection scope inconsistency:** The plan currently references both "final state" and "any edited file" for B2.2. Does it apply to *any* syntax error generated during the execution, or solely the syntax errors persisting in the *final* codebase state?
5. **`target_file` extraction coverage in `bash`:** How reliably can we extract a `target_file` parameter from free-form bash commands to support L1/L2 recovery targeting logic?
6. **Error signal to Liu taxonomy mapping:** We need an explicit, definitive mapping table detailing which `error_signal` patterns map to which Liu et al. categories (e.g., how `TEST_FAILURE` translates directly to B2.1 Logic Error).
7. **Semantic contradiction in "toxic one_shot_success":** In Module B, a trajectory is described as "toxic one_shot_success". However, definitionally, a toxic trajectory (C2) fails Tier 1 verification, while `one_shot_success` implies fully verified. This contradiction must be resolved in the dataset sampling definition.