"""Run the unchanged pixel RLPD learner kernel under a new V5 protocol identity."""

from __future__ import annotations

import scripts.train_rlpd_entropy_ablation as kernel
from scripts.rlpd_entropy_v5_common import STUDY_NAME, read_entropy_v5_protocol


def main():
    kernel.FOLLOWUP_NAME = STUDY_NAME
    kernel.read_entropy_protocol = read_entropy_v5_protocol
    kernel.main()


if __name__ == "__main__":
    main()
