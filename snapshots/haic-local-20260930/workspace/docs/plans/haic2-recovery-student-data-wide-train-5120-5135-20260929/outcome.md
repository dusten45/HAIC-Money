# Infrastructure-invalid pixel-label collection

The registered TRAIN 5120-5135 × tracks 1-3 collection started under plan hash `4cc137d097dac8c1f6a49758194c818764d3d87d0bf29da5a7ab971f61e9df38` but Docker Desktop's Linux engine exited during track 3. Its process handle terminated and the restarted engine had no HAIC experiment container. No final `report.json` was written. Eleven partial per-cell NPZ files remain immutable evidence; they do not satisfy the registered 48-cell coverage or integrity denominator and will not be used for student fitting.

Outcome: `REVISE` for infrastructure failure, not a valid comparable non-improving cycle. The next collection needs a new plan and unused TRAIN cells. No teacher-only lap is a submission-candidate result.
