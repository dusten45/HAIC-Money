"""Read-only steering accounting for the c4e224d crossing source reconstruction.

Only NumPy and the standard library are used. No controller, observation decoder,
or environment is constructed. Attribution is ordered finite-difference accounting
through the actual nonlinear pipeline, not an assertion of independent effects.
"""

import math
import copy
from typing import Any

import numpy as np


RESIDUAL_TOLERANCE = 2e-7
SCHEMA = "koi-steering-terms-v1"
SCHEMA_VERSION = SCHEMA


def observed_center(centers, row):
    """The frozen observed_center extrapolation, including its stable tie order."""
    if row in centers:
        return centers[row]
    known = sorted(centers, key=lambda y: abs(y - row))
    if len(known) < 2:
        return None
    a, b = known[:2]
    return float(np.clip(centers[a] + (row - a) * (centers[b] - centers[a]) / (b - a), 0, 83))


def reconstruct_steering(diagnostics, final_steer, *, state=None):
    """Reconstruct one original crossing decision from its diagnostics/post-act state.

    state keys are obstacle_side, steps, impact_left, impact_left_before_act.
    Optional pre-act impact state takes precedence over the diagnostic same key.
    Archived aliases
    actual_obstacle_side/actual_impact_left and an explicit steps key are accepted.
    Missing inputs remain None; reconstruction never fills missing terms with zero.
    Contributions are ordered road-position, lookahead, avoidance, damping deltas.
    The .55 crossing term replaces .34, including the original different rounding.
    """
    d = diagnostics
    state = state or {}
    centers = {int(k): float(v) for k, v in d.get("road_centers", {}).items()}
    final = float(final_steer)
    side = state.get("obstacle_side", d.get("actual_obstacle_side"))
    steps = state.get("steps", d.get("steps"))
    obj = d.get("near_object")
    correction = d.get("correction")
    impact_before = state.get("impact_left_before_act", d.get("impact_left_before_act"))
    impact_pre_source = (state.get("impact_left_before_act_source", "explicit_state_pre_act")
                         if "impact_left_before_act" in state else
                         d.get("impact_left_before_act_source", "diagnostic_pre_act_capture") if impact_before is not None else None)
    missing = []
    if correction is None:
        missing.append("damping_correction")
    if obj is not None and side is None:
        missing.append("obstacle_side")
    if steps is None:
        missing.append("steps")
    if "road_centers" not in d:
        missing.append("road_centers")
    for key in ("near_object", "contact_active", "geometry_repaired", "far_mode"):
        if key not in d:
            missing.append(key)
    if d.get("contact_mode") != "crossing_projection":
        missing.append("crossing_baseline_mode")
    if not math.isfinite(final) or any(not math.isfinite(v) for v in centers.values()):
        missing.append("finite_final_and_road")
    if correction is not None and not math.isfinite(correction):
        missing.append("finite_damping")
    if side is not None and not math.isfinite(side):
        missing.append("finite_obstacle_side")
    if obj is not None and not all(math.isfinite(v) for v in obj):
        missing.append("finite_near_object")
    if impact_before is not None and (not math.isfinite(impact_before) or impact_before < 0 or impact_before != int(impact_before)):
        missing.append("valid_impact_left_before_act")
    urgency = float(np.clip((obj[0] - 22) / 18, 0, 1)) if obj is not None else 0.0
    near_term = float(side * .34 * urgency) if side is not None and obj is not None else (0.0 if obj is None else None)
    crossing = bool(d.get("contact_active")) and d.get("contact_mode", "crossing_projection") == "crossing_projection"
    projected = d.get("projected_obstacle_x")
    projection_state = ("unavailable" if projected is None else "invalid" if not math.isfinite(projected)
                        else "off" if abs(projected - 42) >= 6 else "crossing")
    base_position = .022 * (centers.get(42, 42.0) - 42)
    base_lookahead = .018 * (centers.get(42, 42.0) - centers.get(54, 42.0))
    middle, near = observed_center(centers, 42), observed_center(centers, 54)
    repaired = bool(d.get("geometry_repaired"))
    raw = dict(base_road_position=base_position if "road_centers" in d else None,
               base_lookahead=base_lookahead if "road_centers" in d else None,
               near_urgency=near_term if "near_object" in d else None,
               crossing_projection=(float(side * .55 * urgency) if crossing and side is not None else None if crossing or "contact_active" not in d else 0.0),
               damping=correction, far_steering=0.0 if d.get("far_mode") == "arrival_speed" else None,
               prefix_steering=0.0 if steps is not None else None)
    result: dict[str, Any] = dict(schema=SCHEMA, raw_terms=raw, stages=[], contributions={},
                  projection_state=projection_state, urgency=urgency if "near_object" in d else None, selected_flank=side, steps=steps,
                  actual_final_steer=final, tolerance=RESIDUAL_TOLERANCE,
                  impact_left_before_act=impact_before, impact_prestate_source=impact_pre_source,
                  identification={}, unidentifiable=missing, inference=[],
                  attribution="ordered nonlinear deltas; avoidance_component is same-pipeline zero-term counterfactual")
    if missing or d.get("far_mode") not in (None, "arrival_speed"):
        result.update(reconstructed_steer=None, residual=None, final_sum_residual=None,
                      reconstruction_valid=False, residual_valid=False, avoidance_component=None,
                      effective_avoidance_raw=None, saturation=None)
        return result
    assert correction is not None and steps is not None and near_term is not None
    correction = float(correction)
    steps = int(steps)
    near_term = float(near_term)
    clip = lambda value: float(np.clip(value, -.7, .7))
    f32 = lambda value: float(np.float32(value))
    inherited = f32(clip(base_position + base_lookahead + near_term))
    damping = f32(clip(inherited + correction))
    complete = 42 in centers and 54 in centers
    parent = d.get("parent_action")
    parent_steer = float(parent[0]) if parent is not None else None
    veto = bool(d.get("braking_proxy_veto", False))
    trigger = bool(d.get("impact_proxy_trigger"))
    eligible_impact = steps > 10 and complete
    impact_ambiguous = False
    impact_assignment = False
    impact_identification = "source_guard_inactive"
    impact_expected = f32(clip(base_position + base_lookahead + correction))
    if eligible_impact:
        if impact_before is not None:
            impact_assignment = impact_before > 0 or trigger
            impact_identification = "original_pre_act_counter_and_trigger"
        elif trigger or d.get("impact_steps_remaining", 0) > 0 or state.get("impact_left", d.get("actual_impact_left", 0)) > 0:
            impact_assignment = True
            impact_identification = "original_trigger_or_positive_post_counter"
        elif crossing and not veto:
            # The source crossing guard excludes a positive pre-counter/trigger.
            impact_identification = "original_crossing_guard_excludes_impact"
        elif repaired or veto:
            impact_ambiguous = True
        else:
            observed_parent = parent_steer if parent_steer is not None else final
            if observed_parent == impact_expected and observed_parent != damping:
                impact_assignment = True
                impact_identification = "unique_parent_float32_replacement"
                result["inference"].append("expired impact hold identified from unique parent_action float32 replacement")
            elif observed_parent == damping and observed_parent != impact_expected:
                impact_identification = "unique_parent_float32_inherited"
            elif (abs(observed_parent - damping) <= RESIDUAL_TOLERANCE
                  and abs(observed_parent - impact_expected) <= RESIDUAL_TOLERANCE):
                impact_ambiguous = True
    # Recovery runs before the coordination veto and preview-row repair.
    impact = impact_assignment and not veto
    if repaired or crossing:
        if middle is None or near is None:
            result["unidentifiable"].append("extrapolated_road")
            result.update(reconstructed_steer=None, residual=None, final_sum_residual=None,
                          reconstruction_valid=False, residual_valid=False, avoidance_component=None,
                          effective_avoidance_raw=None, saturation=None)
            return result
        position = .022 * (middle - 42)
        lookahead = .018 * (middle - near)
    else:
        position, lookahead = base_position, base_lookahead
    effective = float(side * .55 * urgency) if crossing and side is not None else near_term if repaired or not impact else 0.0
    raw.update(selected_road_position=position, selected_lookahead=lookahead,
               impact_replacement_raw=-(near_term) if impact_assignment else 0.0,
               effective_avoidance=effective)

    def pipeline(p, l, a, c):
        if crossing:
            # Contact computes inner clipping in float64, unlike inherited float32.
            return f32(clip(p + l + a + c) if repaired else clip(clip(p + l + a) + c))
        if repaired or impact:
            return f32(clip(p + l + a + c))
        return f32(clip(f32(clip(p + l + a)) + c))

    reconstructed = pipeline(position, lookahead, effective, correction)
    values = [pipeline(0, 0, 0, 0), pipeline(position, 0, 0, 0),
              pipeline(position, lookahead, 0, 0), pipeline(position, lookahead, effective, 0), reconstructed]
    result["contributions"] = dict(zip(("road_position", "lookahead", "avoidance", "damping"),
                                       [b - a for a, b in zip(values, values[1:])]))
    stages = result["stages"]

    def stage(name, before, after, operation):
        stages.append(dict(name=name, before=before, after=after, delta=after - before, operation=operation,
                           saturated=operation == "clip" and abs(after - before) > 1e-12))

    base_raw = base_position + base_lookahead + near_term
    stage("base_clip", base_raw, clip(base_raw), "clip")
    stage("base_float32", clip(base_raw), inherited, "float32")
    stage("damping_add", inherited, inherited + correction, "add")
    stage("damping_clip", inherited + correction, clip(inherited + correction), "clip")
    stage("damping_float32", clip(inherited + correction), damping, "float32")
    selected_parent = damping
    if impact_assignment:
        impact_raw = base_position + base_lookahead + correction
        stage("impact_clip", impact_raw, clip(impact_raw), "clip")
        stage("impact_float32", clip(impact_raw), impact_expected, "float32")
        stage("impact_replace", selected_parent, impact_expected, "replace")
        selected_parent = impact_expected
    if veto:
        stage("impact_veto_replace", selected_parent, damping, "replace")
        selected_parent = damping
    if repaired:
        repair_raw = position + lookahead + near_term + correction
        stage("geometry_repair_clip", repair_raw, clip(repair_raw), "clip")
        stage("geometry_repair_float32", clip(repair_raw), f32(clip(repair_raw)), "float32")
        stage("geometry_repair_replace", selected_parent, f32(clip(repair_raw)), "replace")
        selected_parent = f32(clip(repair_raw))
    if crossing:
        contact_raw = position + lookahead + effective
        if not repaired:
            stage("crossing_inner_clip", contact_raw, clip(contact_raw), "clip")
            contact_raw = clip(contact_raw)
        stage("crossing_outer_clip", contact_raw + correction, clip(contact_raw + correction), "clip")
        stage("crossing_float32", clip(contact_raw + correction), reconstructed, "float32")
        stage("crossing_replace", selected_parent, reconstructed, "replace")
    residual = final - reconstructed
    checkpoints = dict(inherited=None if d.get("inherited_steer") is None else float(d["inherited_steer"]) - inherited,
                       damping=None if d.get("damping_steer") is None else float(d["damping_steer"]) - damping,
                       parent=None if parent_steer is None else parent_steer - selected_parent)
    sum_residual = final - sum(result["contributions"].values())
    valid = abs(residual) <= RESIDUAL_TOLERANCE and all(v is None or abs(v) <= RESIDUAL_TOLERANCE for v in checkpoints.values())
    result.update(reconstructed_steer=reconstructed, residual=residual, final_sum_residual=sum_residual,
                  reconstruction_valid=valid, residual_valid=abs(sum_residual) <= RESIDUAL_TOLERANCE,
                  checkpoint_residuals=checkpoints, unclassified_residual=residual,
                  avoidance_component=reconstructed - pipeline(position, lookahead, 0, correction),
                  effective_avoidance_raw=effective, saturation=any(s["saturated"] for s in stages),
                  impact_assignment=impact_assignment, impact_assignment_ambiguous=impact_ambiguous,
                  float32_rounding=[s for s in stages if s["operation"] == "float32"],
                  active_path="crossing_replacement" if crossing else "geometry_repair" if repaired else "impact_replacement" if impact else "inherited_damping",
                  identification=dict(road="source_formula", avoidance="observed_original_side_and_source_formula",
                                       damping="original_diagnostic", impact=impact_identification,
                                       far="arrival_speed changes pedals only", prefix="first10 skip wrappers, base avoidance still possible"))
    if impact_ambiguous:
        result["unidentifiable"].append("impact_assignment")
        result["identification"]["impact"] = "latent_paths_indistinguishable_without_pre_act_counter"
        result["impact_assignment"] = None
        raw["impact_replacement_raw"] = None
        alternatives = [dict(impact_assignment=False, effective_avoidance_raw=near_term, parent_steer=damping),
                        dict(impact_assignment=True, effective_avoidance_raw=0.0, parent_steer=impact_expected)]
        result["impact_path_alternatives"] = alternatives
        result["stages"].insert(5, dict(name="impact_replace_ambiguous", before=damping, after=impact_expected,
                                       delta=impact_expected - damping, operation="ambiguous_replace", saturated=False,
                                       applied=None, identification="optional recovery assignment not observed"))
        if not crossing and not repaired and not veto:
            result["active_path"] = "ambiguous_impact_or_inherited"
            result["effective_avoidance_raw"] = raw["effective_avoidance"] = None
            result["identification"]["avoidance"] = "raw term ambiguous; common afterclip contribution only"
            # Retain only attributions shared by BOTH compatible source paths.
            other_values = [f32(clip(0)), f32(clip(position)), f32(clip(position + lookahead)),
                            f32(clip(position + lookahead)), f32(clip(position + lookahead + correction))]
            alternative_contributions = dict(zip(("road_position", "lookahead", "avoidance", "damping"),
                                                 [b - a for a, b in zip(other_values, other_values[1:])]))
            result["contribution_ranges"] = {k: [min(v, alternative_contributions[k]), max(v, alternative_contributions[k])]
                                             for k, v in result["contributions"].items()}
            result["contributions"] = {k: v if v == alternative_contributions[k] else None for k, v in result["contributions"].items()}
            if result["avoidance_component"] != 0.0:
                result["avoidance_component"] = None
            if any(v is None for v in result["contributions"].values()):
                result.update(final_sum_residual=None, residual_valid=False)
    return result


def near_scaled_terms(decomposition, alpha, *, final_steer=None):
    """Pure near-term-only counterfactual/actual-postrelease accounting.

    Returns None on unidentified original reconstruction or protected replacement/
    first10 paths. It does NOT select alpha, track objects, certify safety, or act.
    Supply the issued final_steer to verify actual post-release residuals. Omitting
    it labels the output counterfactual. Original terms/diagnostics are not mutated.
    """
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("near alpha must be finite in [0,1]")
    if (decomposition.get("schema") != SCHEMA or not decomposition.get("reconstruction_valid")
            or not decomposition.get("residual_valid") or decomposition.get("unidentifiable")):
        return None
    path = decomposition["active_path"]
    if alpha != 1 and (path not in ("inherited_damping", "geometry_repair") or decomposition.get("steps", 0) <= 10):
        return None
    result = copy.deepcopy(decomposition)
    result.update(near_scale=alpha, reconstruction_context="counterfactual" if final_steer is None else "actual_postrelease",
                  baseline_reconstructed_steer=decomposition["reconstructed_steer"])
    if alpha == 1:
        expected = decomposition["reconstructed_steer"]
    else:
        raw = result["raw_terms"]
        p, l, correction = (raw[k] for k in ("selected_road_position", "selected_lookahead", "damping"))
        near = raw["near_urgency"] * alpha
        raw.update(near_urgency_unscaled=raw["near_urgency"], near_urgency=near, effective_avoidance=near)
        clip = lambda value: float(np.clip(value, -.7, .7))
        f32 = lambda value: float(np.float32(value))

        def pipeline(p, l, a, c):
            return f32(clip(p + l + a + c)) if path == "geometry_repair" else f32(clip(f32(clip(p + l + a)) + c))

        expected = pipeline(p, l, near, correction)
        values = [pipeline(0, 0, 0, 0), pipeline(p, 0, 0, 0), pipeline(p, l, 0, 0), pipeline(p, l, near, 0), expected]
        result["contributions"] = dict(zip(("road_position", "lookahead", "avoidance", "damping"), [b - a for a, b in zip(values, values[1:])]))
        result["avoidance_component"] = expected - pipeline(p, l, 0, correction)
        result["effective_avoidance_raw"] = near
        result["baseline_stages"] = result["stages"]
        stages = []
        first = p + l + near
        if path == "geometry_repair":
            operations = [("geometry_release_clip", first + correction, clip(first + correction), "clip"),
                          ("geometry_release_float32", clip(first + correction), expected, "float32")]
        else:
            inherited = f32(clip(first))
            operations = [("near_release_inner_clip", first, clip(first), "clip"),
                          ("near_release_inner_float32", clip(first), inherited, "float32"),
                          ("near_release_damping_add", inherited, inherited + correction, "add"),
                          ("near_release_outer_clip", inherited + correction, clip(inherited + correction), "clip"),
                          ("near_release_outer_float32", clip(inherited + correction), expected, "float32")]
        for name, before, after, operation in operations:
            stages.append(dict(name=name, before=before, after=after, delta=after - before, operation=operation,
                               saturated=operation == "clip" and abs(after - before) > 1e-12))
        result.update(stages=stages, saturation=any(s["saturated"] for s in stages),
                      float32_rounding=[s for s in stages if s["operation"] == "float32"],
                      checkpoint_residuals={}, baseline_checkpoint_residuals=decomposition["checkpoint_residuals"])
        result["identification"]["avoidance"] = "near-term-only source-pipeline replay; safety/lifecycle external and unassessed"
    actual = expected if final_steer is None else float(final_steer)
    residual = actual - expected
    result.update(reconstructed_steer=expected, actual_final_steer=actual, residual=residual, unclassified_residual=residual,
                  final_sum_residual=actual - sum(result["contributions"].values()),
                  reconstruction_valid=math.isfinite(actual) and abs(residual) <= RESIDUAL_TOLERANCE,
                  residual_valid=math.isfinite(actual) and abs(actual - sum(result["contributions"].values())) <= RESIDUAL_TOLERANCE)
    return result


def near_scaled_steer(decomposition, alpha):
    """Scalar convenience API; None means retain the original command."""
    terms = near_scaled_terms(decomposition, alpha)
    return None if terms is None else terms["reconstructed_steer"]


def decompose(driver, observation, action, diagnostics=None):
    """Post-act passive wrapper API; driver is the ORIGINAL crossing instance."""
    del observation
    diagnostics = driver.last_step_diagnostics() if diagnostics is None else diagnostics
    return reconstruct_steering(diagnostics, float(action[0]), state={
        "obstacle_side": float(driver.base._obstacle_side), "steps": int(driver.steps),
        "impact_left": int(driver.impact_left),
    })
