"""Run the frozen one-factor orchestration kernel with a new V5 source identity."""

from __future__ import annotations

import subprocess

import scripts.project_rlpd_entropy_screen_receipts as projector
import scripts.run_rlpd_entropy_ablation as kernel
from scripts.rlpd_entropy_v5_common import STUDY_NAME, read_entropy_v5_protocol


def main():
    kernel.STUDY_NAME = STUDY_NAME
    kernel.read_entropy_protocol = read_entropy_v5_protocol
    projector.STUDY_NAME = STUDY_NAME
    projector.read_entropy_protocol = read_entropy_v5_protocol
    original_run = subprocess.run

    def run_v5_training(command, *args, **kwargs):
        if (
            isinstance(command, list)
            and len(command) >= 3
            and command[1:3] == ["-m", "scripts.train_rlpd_entropy_ablation"]
        ):
            command = list(command)
            command[2] = "scripts.train_rlpd_entropy_v5"
        return original_run(command, *args, **kwargs)

    kernel.subprocess.run = run_v5_training
    kernel.main()


if __name__ == "__main__":
    main()
