# Pixel speed student first TUNE attempt: infrastructure invalid

The registered run started control cell 1:5200, which finished in 21.96 seconds. The student failed to load its checkpoint before its first action because the TRAIN artifact stored normalization arrays as NumPy objects and the runtime requested `torch.load(weights_only=True)`. This is an infrastructure failure, not a driving result. The v2 outcome is `REVISE → GATE_REVIEW_REVISE → STOPPED`, no release. The original run and checkpoint remain unchanged.

A new exact plan may retry the same registered TUNE block after a loader-only repair, with a new plan hash and three local approval records. The retry cannot be described as an independent fresh block relative to this attempt. No candidate lap or completion claim can be made from this run.
