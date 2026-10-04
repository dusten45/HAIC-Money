"""Bounded consumed-trace replay; privileged diagnostics never enter Agent.

Reproduce the frozen combined track1 prefix exactly, stopping at its first
contact (at most100 actions). Record each controller's actual single proposal,
current camera geometry and post-action motion. This is not a new lap screen.
"""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from local_simulator.environment import create_environment, reset_environment
from local_simulator.schema import MapSpec


def main():
    source=Path("agents/apex_2026/fast_combined_agent.py")
    trace_path=Path(".haic-artifacts/apex-speed-20261005/combined-v1-traces/track-1-seed-516237.json")
    output=Path(".haic-artifacts/apex-speed-preview-combined-contact")
    output.mkdir(exist_ok=False)
    trace=json.loads(trace_path.read_text())
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    if source_hash!="31059f6d3e4906199d8a703aa46e9c5d76df529ef207296d61f6f7c1bf24ce52":
        raise ValueError("diagnostic requires the exact frozen combined V1 source")
    spec=importlib.util.spec_from_file_location("preview_combined_contact_submission",source)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    agent=module.Agent()
    proposals={}

    def capture(name,controller):
        original=controller.act
        def wrapped(observation):
            action=original(observation)
            proposals[name]={"action":action.tolist(),"mode":controller.mode,
                "speed":float(controller.last_speed),"target":float(controller.last_target),
                "yaw":float(controller.last_yaw),"steer":float(controller.last_steer)}
            return action
        controller.act=wrapped

    capture("fast",agent.fast)
    capture("hazard",agent.hazard)
    map_spec=MapSpec(1,516237,"official",(),700,4)
    environment,raw=create_environment(map_spec,render_mode=None)
    rows=[]
    frames=[]
    planned_world=None
    plans={}

    def actual():
        h=raw.car.hull
        angle=float(h.angle)
        v=np.asarray(h.linearVelocity,dtype=float)
        right=np.asarray([np.cos(angle),np.sin(angle)])
        forward=np.asarray([-np.sin(angle),np.cos(angle)])
        return {"position":list(h.position),"angle":angle,
            "speed_mps":float(np.linalg.norm(v)),"camera_yaw_radps":-float(h.angularVelocity),
            "lateral_speed_mps":float(v@right),"forward_speed_mps":float(v@forward),
            "sideslip_rad":float(np.arctan2(v@right,v@forward)),
            "wheel_angles": [float(w.joint.angle) for w in raw.car.wheels[:2]],
            "wheel_contacts": [bool(w.tiles) for w in raw.car.wheels]}

    try:
        observation,_=reset_environment(environment,map_spec)
        agent.reset(observation)
        for step,saved in enumerate(trace[:100]):
            before=actual()
            action=agent.act(observation)
            np.testing.assert_array_equal(action,np.asarray(saved["action"],np.float32))
            hazard=agent.hazard
            frame=hazard._metric._frame(observation)
            road=hazard._metric._road(frame)
            bbox=hazard._robust._corridor_bbox
            plan=hazard.last_pass_plan
            camera={"bbox":bbox,"commit_rejection":hazard.last_commit_rejection}
            if road is not None:
                ahead,side,_,lefts,rights=road
                camera.update(road_horizon=float(ahead[-1]),
                    road_ego_side=float(np.interp(0.,ahead,side)),
                    road_support_gap_max=float(np.diff(ahead).max()))
                if bbox is not None:
                    circle=hazard._circle_center(bbox)
                    fit=(ahead>=-2.) & (ahead<=circle[1]+6.)
                    camera.update(circle=circle.tolist(),road_fit_samples=int(fit.sum()))
                    if fit.sum()>=4:
                        q=np.polyfit(ahead[fit],side[fit],3)
                        camera["road_fit_residual_max"]=float(abs(np.polyval(q,ahead[fit])-side[fit]).max())
                    angle=before["angle"]
                    right=np.asarray([np.cos(angle),np.sin(angle)])
                    forward=np.asarray([-np.sin(angle),np.cos(angle)])
                    ego=np.asarray(before["position"])
                    truth=[]
                    for body in raw.obstacles:
                        delta=np.asarray(body.position)-ego
                        point=np.asarray([delta@right,delta@forward])
                        truth.append((float(np.linalg.norm(point-circle)),point.tolist(),float(body.fixtures[0].shape.radius)))
                    if truth:
                        error,position,radius=min(truth)
                        camera["privileged_nearest_circle"]={"camera_coord":position,"radius":radius,"camera_center_error_m":error}
            if plan is not None:
                camera["plan"]={k:float(plan[k]) for k in
                    ("side","road_margin","obstacle_margin","min_separation","max_curvature","obstacle_distance")}
                camera["plan"].update(ego_reference_x=float(plan["x"][0]),
                    ego_reference_heading=float(plan["heading"][0]))
                angle=before["angle"]
                right=np.asarray([np.cos(angle),np.sin(angle)])
                forward=np.asarray([-np.sin(angle),np.cos(angle)])
                planned_world=(np.asarray(before["position"])+
                    plan["x"][:,None]*right+plan["y"][:,None]*forward)
                plans[str(step)]={"x":plan["x"].tolist(),"y":plan["y"].tolist(),
                    "heading":plan["heading"].tolist(),"world":planned_world.tolist()}
            observation,_,terminated,truncated,info=environment.step(action)
            after=actual()
            row={"step":step,"selected":agent.mode,"action":action.tolist(),
                "proposals":dict(proposals),"camera":camera,"before":before,"after":after,
                "collision":bool(info.get("collision",False)),"progress":float(info.get("progress",0.))}
            if planned_world is not None:
                position=np.asarray(after["position"])
                p=planned_world[:-1];d=np.diff(planned_world,axis=0)
                fractions=np.clip(np.sum((position-p)*d,axis=1)/np.sum(d*d,axis=1),0.,1.)
                projected=p+fractions[:,None]*d
                index=int(np.argmin(np.linalg.norm(projected-position,axis=1)))
                row["post_action_reference_error_m"]=float(np.linalg.norm(projected[index]-position))
            rows.append(row)
            if step>=45:
                frames.append(frame.copy())
            if row["collision"] or terminated or truncated:
                break
    finally:
        environment.close()
    assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
    np.savez_compressed(output/"frames.npz",frames=np.asarray(frames),steps=np.arange(45,45+len(frames)))
    report={"classification":"consumed combinedV1 exact action prefix diagnostic; not fresh validation",
        "source_sha256":source_hash,"trace_sha256":hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        "track_id":1,"seed":516237,"max_actions":100,"exact_actions":len(rows),
        "stopped_at_first_contact":rows[-1]["collision"],"rows":rows,"plans":plans,
        "privileged_diagnostics": "actual motion, wheel contacts and nearest official circle; no privileged fields are passed to inference"}
    (output/"diagnostic.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"source_sha256":source_hash,"exact_actions":len(rows),
        "first_contact_step":rows[-1]["step"],"output":str(output)}))


if __name__=="__main__":
    main()
