import json
from pathlib import Path

import pytest

from research_ops.index import (
    discover_result_files,
    normalize_json_record,
    scan_workspace,
    sync_results_document,
)
from research_ops.orchestrator import build_improvement_plan
from research_ops.policy import compare_records, promotion_gate


def test_normalize_json_record_extracts_strategy_and_metrics(tmp_path: Path):
    source = tmp_path / "task5-eval-fast-fullcap" / "summary.json"
    source.parent.mkdir()
    source.write_text(
        json.dumps(
            {
                "by_mode": {
                    "ppo_only": {
                        "episodes": 10,
                        "completed": 3,
                        "completion_rate": 0.3,
                        "mean_progress": 0.41,
                        "median_finished_lap_ms": 21000,
                        "p90_finished_lap_ms": 26000,
                        "collisions": 2,
                        "damage": 1,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    records = normalize_json_record(source)

    assert len(records) == 1
    assert records[0].strategy == "ppo_actor_only"
    assert records[0].split == "held_out"
    assert records[0].completion_rate == pytest.approx(0.3)
    assert records[0].p90_finished_lap_ms == pytest.approx(26000)


def test_normalize_speed_target_summary_emits_per_target_tune_records(tmp_path: Path):
    source = tmp_path / "ppo-speed-target" / "summary.json"
    source.parent.mkdir()
    episodes = [
        {
            "speed_target_m_s": 70,
            "finished": False,
            "progress": 0.4,
            "collision": True,
            "final_damage": 0.2,
            "actor_inference_latency_ms": {"p95": 5.5},
        },
        {
            "speed_target_m_s": 70,
            "finished": False,
            "progress": 0.6,
            "collision": False,
            "final_damage": 0.0,
            "actor_inference_latency_ms": {"p95": 5.7},
        },
        {
            "speed_target_m_s": 84,
            "finished": True,
            "progress": 1.0,
            "collision": False,
            "final_damage": 0.0,
            "lap_time_s": 12.5,
            "actor_inference_latency_ms": {"p95": 5.2},
        },
        {
            "speed_target_m_s": 84,
            "finished": False,
            "progress": 0.7,
            "collision": True,
            "final_damage": 0.2,
            "actor_inference_latency_ms": {"p95": 5.9},
        },
    ]
    source.write_text(
        json.dumps(
            {
                "experiment": "ppo-speed-target-lr1e4",
                "split": "tune",
                "episodes": episodes,
                "target_summaries": {
                    "target_70": {
                        "episode_count": 2,
                        "finish_count": 0,
                        "finish_rate": 0.0,
                        "collision_episodes": 1,
                        "mean_final_damage": 0.1,
                        "actor_inference_p95_ms_max": 5.7,
                    },
                    "target_84": {
                        "episode_count": 2,
                        "finish_count": 1,
                        "finish_rate": 0.5,
                        "collision_episodes": 1,
                        "mean_final_damage": 0.1,
                        "median_finished_lap_time_s": 12.5,
                        "p90_finished_lap_time_s": 12.5,
                        "actor_inference_p95_ms_max": 5.9,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    records = normalize_json_record(source)

    assert len(records) == 2
    assert records[0].split == "tune"
    assert all(record.strategy == "ppo_actor_only" for record in records)
    assert len({record.record_id for record in records}) == 2
    target70 = next(record for record in records if record.variant == "target_70")
    target84 = next(record for record in records if record.variant == "target_84")
    assert target70.episodes == 2
    assert target70.completion_rate == pytest.approx(0.0)
    assert target70.mean_progress == pytest.approx(0.5)
    assert target70.collisions == pytest.approx(1.0)
    assert target70.damage == pytest.approx(0.1)
    assert target70.act_latency_p95_ms == pytest.approx(5.7)
    assert target84.episodes == 2
    assert target84.completion_rate == pytest.approx(0.5)
    assert target84.mean_progress == pytest.approx(0.85)
    assert target84.median_finished_lap_ms == pytest.approx(12500)
    assert target84.p90_finished_lap_ms == pytest.approx(12500)


def test_normalize_arm_summary_emits_per_arm_tune_records(tmp_path: Path):
    source = tmp_path / "ppo-bc-reference-kl" / "evaluation-u8" / "summary.json"
    source.parent.mkdir(parents=True)
    episodes = [
        {
            "arm": "ppo_control_kl0",
            "training_seed": 8104,
            "finished": False,
            "valid_under_13": False,
            "progress": 0.38,
            "collision": True,
            "collision_decisions": 5,
            "final_damage": 1.0,
            "mean_speed": 36.9,
            "actor_inference_latency_ms": {"p95": 7.3},
        },
        {
            "arm": "ppo_bc_reference_kl_0p5",
            "training_seed": 8104,
            "finished": True,
            "valid_under_13": False,
            "lap_time_s": 19.6,
            "progress": 1.0,
            "collision": True,
            "collision_decisions": 3,
            "final_damage": 0.6,
            "mean_speed": 42.4,
            "actor_inference_latency_ms": {"p95": 7.2},
        },
    ]
    source.write_text(
        json.dumps(
            {
                "experiment": "ppo-bc-reference-kl-speed70-lr1e4-v1",
                "split": "tune",
                "runtime_policy": "ppo_actor_only",
                "episodes": episodes,
                "arm_summaries": {
                    "ppo_control_kl0": {
                        "episode_count": 1,
                        "finish_count": 0,
                        "finish_rate": 0.0,
                        "valid_under_13_fraction": 0.0,
                        "collision_episodes": 1,
                        "mean_final_damage": 1.0,
                        "mean_dnf_progress": 0.38,
                        "actor_inference_p95_ms_max": 7.3,
                    },
                    "ppo_bc_reference_kl_0p5": {
                        "episode_count": 1,
                        "finish_count": 1,
                        "finish_rate": 1.0,
                        "valid_under_13_fraction": 0.0,
                        "collision_episodes": 1,
                        "mean_final_damage": 0.6,
                        "median_finished_lap_time_s": 19.6,
                        "p90_finished_lap_time_s": 19.6,
                        "actor_inference_p95_ms_max": 7.2,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    records = normalize_json_record(source)

    assert len(records) == 2
    assert {record.split for record in records} == {"tune"}
    assert all(record.strategy == "ppo_actor_only" for record in records)
    assert len({record.record_id for record in records}) == 2
    control = next(record for record in records if record.variant == "ppo_control_kl0")
    treatment = next(record for record in records if record.variant == "ppo_bc_reference_kl_0p5")
    assert control.episodes == 1
    assert control.completion_rate == pytest.approx(0.0)
    assert control.mean_progress == pytest.approx(0.38)
    assert control.collisions == pytest.approx(1.0)
    assert control.damage == pytest.approx(1.0)
    assert control.restriction_status == "unknown"
    assert treatment.episodes == 1
    assert treatment.completion_rate == pytest.approx(1.0)
    assert treatment.valid_under_13_fraction == pytest.approx(0.0)
    assert treatment.mean_progress == pytest.approx(1.0)
    assert treatment.collisions == pytest.approx(1.0)
    assert treatment.damage == pytest.approx(0.6)
    assert treatment.median_finished_lap_ms == pytest.approx(19600)
    assert treatment.p90_finished_lap_ms == pytest.approx(19600)
    assert treatment.act_latency_p95_ms == pytest.approx(7.2)
    assert treatment.restriction_status == "unknown"


def test_scan_workspace_preserves_arm_records_and_under_13_metric(tmp_path: Path):
    source = tmp_path / "artifacts" / "haic" / "ppo-bc-reference-kl" / "summary.json"
    source.parent.mkdir(parents=True)
    source.write_text(
        json.dumps(
            {
                "experiment": "ppo-bc-reference-kl",
                "split": "tune",
                "episodes": [
                    {
                        "arm": "ppo_bc_reference_kl_0p5",
                        "finished": True,
                        "valid_under_13": False,
                        "progress": 1.0,
                        "lap_time_s": 19.6,
                    }
                ],
                "arm_summaries": {
                    "ppo_bc_reference_kl_0p5": {
                        "episode_count": 1,
                        "finish_count": 1,
                        "finish_rate": 1.0,
                        "valid_under_13_fraction": 0.0,
                        "collision_episodes": 1,
                        "mean_final_damage": 0.6,
                        "median_finished_lap_time_s": 19.6,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    records = scan_workspace(tmp_path)

    assert len(records) == 1
    assert records[0].variant == "ppo_bc_reference_kl_0p5"
    assert records[0].valid_under_13_fraction == pytest.approx(0.0)


def test_compare_records_follows_competition_ordering():
    slower_but_complete = {
        "record_id": "a",
        "strategy": "ppo_actor_only",
        "split": "held_out",
        "episodes": 8,
        "completion_rate": 1.0,
        "median_finished_lap_ms": 30000,
        "p90_finished_lap_ms": 32000,
        "mean_progress": 1.0,
        "collisions": 3,
        "damage": 0,
        "restriction_status": "pass",
    }
    faster_but_incomplete = {
        **slower_but_complete,
        "record_id": "b",
        "strategy": "ppo_cem",
        "completion_rate": 0.75,
        "median_finished_lap_ms": 18000,
        "p90_finished_lap_ms": 20000,
        "mean_progress": 0.82,
    }

    ranked = compare_records([slower_but_complete, faster_but_incomplete])

    assert [row["strategy"] for row in ranked] == ["ppo_actor_only", "ppo_cem"]


def test_promotion_gate_rejects_restriction_violation():
    candidate = {
        "strategy": "ppo_cem",
        "split": "held_out",
        "episodes": 8,
        "completion_rate": 1.0,
        "median_finished_lap_ms": 19000,
        "p90_finished_lap_ms": 21000,
        "mean_progress": 1.0,
        "collisions": 0,
        "damage": 0,
        "restriction_status": "fail",
    }
    sota = {
        **candidate,
        "strategy": "ppo_actor_only",
        "median_finished_lap_ms": 22000,
        "p90_finished_lap_ms": 25000,
        "restriction_status": "pass",
    }

    decision = promotion_gate(candidate, sota)

    assert decision.promote is False
    assert "restriction" in decision.reason.lower()


def test_promotion_gate_requires_package_smoke_for_submission_candidate():
    candidate = {
        "strategy": "ppo_cem",
        "split": "held_out",
        "episodes": 8,
        "completion_rate": 1.0,
        "median_finished_lap_ms": 19000,
        "p90_finished_lap_ms": 21000,
        "mean_progress": 1.0,
        "collisions": 0,
        "damage": 0,
        "restriction_status": "pass",
    }
    sota = {
        **candidate,
        "strategy": "ppo_actor_only",
        "median_finished_lap_ms": 22000,
        "p90_finished_lap_ms": 25000,
        "package_smoke_status": "pass",
    }

    decision = promotion_gate(candidate, sota)

    assert decision.promote is False
    assert "smoke" in decision.reason.lower()


def test_sync_results_document_replaces_only_generated_section(tmp_path: Path):
    target = tmp_path / "RESULTS.md"
    target.write_text(
        "# Results\n\nHuman notes stay.\n\n<!-- BEGIN GENERATED RESULTS -->\nold\n<!-- END GENERATED RESULTS -->\n",
        encoding="utf-8",
    )
    records = [
        {
            "record_id": "abc123",
            "strategy": "ppo_actor_only",
            "split": "held_out",
            "episodes": 8,
            "completion_rate": 0.75,
            "mean_progress": 0.78,
            "median_finished_lap_ms": 19320,
            "p90_finished_lap_ms": 22420,
            "collisions": 0,
            "damage": 0,
            "restriction_status": "pass",
            "source": "artifacts/example.json",
        }
    ]

    sync_results_document(target, records)
    content = target.read_text(encoding="utf-8")

    assert "Human notes stay." in content
    assert "abc123" in content
    assert "old" not in content


def test_discover_result_files_ignores_debug_traces_and_checkpoints(tmp_path: Path):
    report = tmp_path / "artifacts" / "haic" / "run-a" / "summary.json"
    trace = tmp_path / "artifacts" / "haic" / "run-a" / "decision_trace.json"
    checkpoint = tmp_path / "artifacts" / "haic" / "run-a" / "policy.pt"
    report.parent.mkdir(parents=True)
    report.write_text("{}", encoding="utf-8")
    trace.write_text("{}", encoding="utf-8")
    checkpoint.write_bytes(b"weights")

    files = discover_result_files(tmp_path)

    assert files == [report]


def test_build_improvement_plan_requires_explicit_submission(tmp_path: Path):
    for name in ("RULES.md", "SOTA.md", "RESULTS.md", "COMPETITION_INFO.md", "RESTRICTIONS.md", "report.pdf"):
        (tmp_path / name).write_bytes(b"ready")

    plan = build_improvement_plan(tmp_path, "성능 개선해 줘", records=[])

    assert plan["trigger"] == "성능 개선해 줘"
    assert set(plan["strategy_lanes"]) == {
        "ppo_actor_only",
        "ppo_cem",
        "vision_corridor_teacher",
    }
    assert plan["submission_policy"]["requires_explicit_confirmation"] is True


def test_build_improvement_plan_exposes_promotion_decisions(tmp_path: Path):
    for name in ("RULES.md", "SOTA.md", "RESULTS.md", "COMPETITION_INFO.md", "RESTRICTIONS.md", "report.pdf"):
        (tmp_path / name).write_bytes(b"ready")
    records = [
        {
            "record_id": "baseline",
            "strategy": "ppo_actor_only",
            "split": "held_out",
            "episodes": 8,
            "completion_rate": 0.75,
            "mean_progress": 0.78,
            "median_finished_lap_ms": 19320,
            "p90_finished_lap_ms": 22420,
            "collisions": 0,
            "damage": 0,
            "restriction_status": "pass",
            "package_smoke_status": "pass",
        },
        {
            "record_id": "cem",
            "strategy": "ppo_cem",
            "split": "held_out",
            "episodes": 8,
            "completion_rate": 1.0,
            "mean_progress": 1.0,
            "median_finished_lap_ms": 18000,
            "p90_finished_lap_ms": 20000,
            "collisions": 0,
            "damage": 0,
            "restriction_status": "pass",
        },
    ]

    plan = build_improvement_plan(tmp_path, "성능 개선해 줘", records=records)

    assert plan["promotion"]["ppo_cem"]["promote"] is False
    assert "smoke" in plan["promotion"]["ppo_cem"]["reason"].lower()
