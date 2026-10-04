"""Camera-only offline C2 path and speed-envelope diagnostic, not inference."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


def quintic_coefficients(length,x0,slope0,second0,x1,slope1,second1):
    """Normalized y/length power coefficients matching six C2 conditions."""
    c=np.zeros(6)
    c[:3]=x0,slope0*length,.5*second0*length**2
    rhs=np.asarray([x1-c[:3].sum(),slope1*length-c[1]-2*c[2],
                    second1*length**2-2*c[2]])
    c[3:]=np.linalg.solve(np.asarray([[1.,1.,1.],[3.,4.,5.],[6.,12.,20.]]),rhs)
    return c


def braking_capacity(speed,curvature):
    """Illustrative shared190m/s² circle and100m/s² longitudinal budget."""
    lateral=speed*speed*abs(curvature)
    return float(min(100.,np.sqrt(max(0.,190.**2-lateral**2))))


def speed_envelope(arc,curvature,speed,previous_steer):
    """Fixed reaction, joint/rate caps and shared-force braking reachability.

    This scalar circle is a diagnostic model, not the actual four-tire force
    solver. Conservatively use the larger neighboring curvature in each
    segment. Numerical bisection solves a bound, not a policy parameter grid.
    """
    arc=np.asarray(arc,dtype=float)
    curvature=np.asarray(curvature,dtype=float)
    steering=np.arctan(3.24*curvature)
    steering_gradient=abs(np.gradient(steering,arc))
    lateral_cap=np.sqrt(190./np.maximum(abs(curvature),.0001))
    motor_cap=3./np.maximum(steering_gradient,.0001)
    point_cap=np.minimum(100.,np.minimum(lateral_cap,motor_cap))
    backward=np.empty(len(arc))
    backward[-1]=min(85.,point_cap[-1])
    for i in range(len(arc)-2,-1,-1):
        gap=arc[i+1]-arc[i]
        k=max(abs(curvature[i]),abs(curvature[i+1]))
        low,high=0.,float(point_cap[i])
        for _ in range(36):
            v=.5*(low+high)
            if v*v <= backward[i+1]**2+2*braking_capacity(v,k)*gap:
                low=v
            else:
                high=v
        backward[i]=low
    reaction=.08*speed
    reaction_points=arc<=reaction
    # Include the interpolated endpoint of the reaction interval.
    reaction_cap=min(float(point_cap[reaction_points].min()),
                     float(np.interp(min(reaction,arc[-1]),arc,point_cap)))
    after_reaction_cap=float(np.interp(min(reaction,arc[-1]),arc,backward))
    lateral_reaction=max(float(abs(curvature[reaction_points]).max()),
                         abs(float(np.interp(min(reaction,arc[-1]),arc,curvature))))*speed**2
    initial_slew=abs(steering[0]-previous_steer)
    joint_ok=bool(abs(steering).max()<=.4)
    slew_ok=bool(initial_slew<=.24)
    forward=np.full(len(arc),speed)
    for i in range(len(arc)-1):
        ds=max(0.,arc[i+1]-max(reaction,arc[i]))
        k=max(abs(curvature[i]),abs(curvature[i+1]))
        forward[i+1]=np.sqrt(max(0.,forward[i]**2-2*braking_capacity(forward[i],k)*ds))
    violations=np.flatnonzero(forward>point_cap+1e-6)
    feasible=bool(joint_ok and slew_ok and reaction<arc[-1]
                  and speed<=reaction_cap+1e-6 and speed<=after_reaction_cap+1e-6)
    return {"reaction_distance_m":reaction,"initial_lateral_ok":bool(speed<=lateral_cap[0]),
        "reaction_lateral_ok":bool(lateral_reaction<=190.+1e-6),
        "max_reaction_lateral_accel_mps2":lateral_reaction,
        "joint_ok":joint_ok,"initial_action_slew_ok":slew_ok,
        "initial_steer_change_rad":float(initial_slew),
        "reaction_motor_and_lateral_cap_mps":reaction_cap,
        "after_reaction_braking_entry_cap_mps":after_reaction_cap,
        "current_hud_speed_mps":speed,"feasible_under_model":feasible,
        "first_max_braking_violation_arc_m":float(arc[violations[0]]) if len(violations) else None,
        "point_speed_cap_mps":point_cap.tolist(),"backward_speed_cap_mps":backward.tolist(),
        "max_braking_forward_speed_mps":forward.tolist()}


def geometry(agent,road,x,y,slope,curvature,circle,uncertainty):
    heading=np.arctan(slope)
    ahead,_,_,lefts,rights=road
    lefts,rights=(lefts-42.)/1.3608,(rights-42.)/1.3608
    road_margins=[agent._hull_road_margin(py,px,ph,ahead,lefts,rights)
                  for px,py,ph in zip(x,y,heading)]
    missing=[float(y[i]) for i,v in enumerate(road_margins) if v is None]
    known=[v for v in road_margins if v is not None]
    dx,dy=np.diff(x),np.diff(y)
    fraction=np.clip(((circle[0]-x[:-1])*dx+(circle[1]-y[:-1])*dy)/(dx*dx+dy*dy),0.,1.)
    separation=float(np.hypot(x[:-1]+fraction*dx-circle[0],y[:-1]+fraction*dy-circle[1]).min())
    cx,cy=circle[0]-x,circle[1]-y
    lateral=cx*np.cos(heading)-cy*np.sin(heading)
    longitudinal=cx*np.sin(heading)+cy*np.cos(heading)
    outside_lat=np.maximum(abs(lateral)-1.6,0.)
    outside_long=np.maximum(np.maximum(longitudinal-2.6,-longitudinal-2.4),0.)
    margin=float((np.hypot(outside_lat,outside_long)-1.2).min())
    uncertain_margin=margin-uncertainty
    nominal=bool(not missing and min(known)>=.30 and margin>=.75
                 and separation>=3.5 and abs(curvature).max()<=np.tan(.4)/3.24)
    return {"known_road_margin_min_m":float(min(known)) if known else None,
        "full_road_support":not bool(missing),"unsupported_pose_y_m":missing,
        "circle_hull_margin_m":margin,"circle_hull_margin_with_center_uncertainty_m":uncertain_margin,
        "centerline_separation_m":separation,"max_curvature_per_m":float(abs(curvature).max()),
        "nominal_geometry_ok":nominal,"one_pixel_uncertainty_geometry_ok":bool(nominal and uncertain_margin>=.75)}


def construction(road,circle,yaw,speed,kind):
    ahead,side,*_=road
    end=circle[1]+3.
    fit=(ahead>=-2.) & (ahead<=end+3.)
    q=np.polyfit(ahead[fit],side[fit],3)
    derivative=np.polyder(q)
    second=np.polyder(derivative)
    residual=float(abs(np.polyval(q,ahead[fit])-side[fit]).max())
    goal=circle[0]-4.*np.sqrt(1.+np.polyval(derivative,circle[1])**2)-np.polyval(q,circle[1])
    switch=max(3.,circle[1]-4.)
    y=np.unique(np.r_[np.arange(0.,end,.35),switch,end])
    t=np.minimum(1.,y/switch)
    if kind=="hermite":
        initial=-float(np.polyval(q,0.));initial_derivative=-float(np.polyval(derivative,0.))
        h00,h10,h01=2*t**3-3*t**2+1,t**3-2*t**2+t,-2*t**3+3*t**2
        hd00,hd10,hd01=6*t*t-6*t,3*t*t-4*t+1,-6*t*t+6*t
        hdd00,hdd10,hdd01=12*t-6,6*t-4,-12*t+6
        x=np.polyval(q,y)+h00*initial+h10*switch*initial_derivative+h01*goal
        slope=np.polyval(derivative,y)+(hd00*initial+hd10*switch*initial_derivative+hd01*goal)/switch
        bend=np.polyval(second,y)+(hdd00*initial+hdd10*switch*initial_derivative+hdd01*goal)/switch**2
    else:
        coefficients=quintic_coefficients(switch,0.,0.,yaw/max(speed,1.),
            float(np.polyval(q,switch)+goal),float(np.polyval(derivative,switch)),float(np.polyval(second,switch)))
        first=np.polynomial.polynomial.polyder(coefficients)
        next_derivative=np.polynomial.polynomial.polyder(first)
        x=np.polynomial.polynomial.polyval(t,coefficients)
        slope=np.polynomial.polynomial.polyval(t,first)/switch
        bend=np.polynomial.polynomial.polyval(t,next_derivative)/switch**2
    following=y>=switch
    x[following]=np.polyval(q,y[following])+goal
    slope[following]=np.polyval(derivative,y[following])
    bend[following]=np.polyval(second,y[following])
    curvature=bend/(1+slope*slope)**1.5
    arc=np.r_[0.,np.cumsum(np.hypot(np.diff(x),np.diff(y)))]
    return x,y,slope,curvature,arc,{"join_y_m":switch,"end_y_m":end,"road_fit_residual_max_m":residual,
        "road_fit_points":int(fit.sum()),"join_position_x_m":float(np.polyval(q,switch)+goal),
        "join_slope":float(np.polyval(derivative,switch)),"join_second_derivative":float(np.polyval(second,switch))}


def main():
    source=Path("agents/apex_2026/fast_combined_agent.py")
    source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
    if source_sha!="31059f6d3e4906199d8a703aa46e9c5d76df529ef207296d61f6f7c1bf24ce52":
        raise ValueError("camera study requires exact frozen combinedV1 source")
    fixture=Path("agents/apex_2026/tests/fixtures/combined_v1_entry_camera.npz")
    data=np.load(fixture)
    spec=importlib.util.spec_from_file_location("quintic_frozen_camera_helpers",source)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    agent=module.Agent()
    trace_path=Path(".haic-artifacts/apex-speed-20261005/combined-v1-traces/track-1-seed-516237.json")
    trace=json.loads(trace_path.read_text())
    uncertainty=float(np.hypot(1./1.3608,1./1.701))
    rows=[]
    for step in (58,59):
        idx=int(np.flatnonzero(data["steps"]==step)[0]);frame=data["frames"][idx]
        road=agent.hazard._metric._road(frame)
        speed=agent.hazard._metric._speed(frame);yaw=agent.hazard._camera_yaw(frame)
        if step==58:
            circles=agent.fast._circles(frame,agent.fast._road(frame))
            distance,lateral=circles[0];circle=np.array([lateral,distance])
            # Floating metric bbox from camera centroid and known official
            # radius only. No world pose or map is used in this study.
            cx,cy=42.+1.3608*lateral,63.-1.701*distance
            bbox=[cx-1.3608*1.2,cy-1.701*1.2,cx+1.3608*1.2,cy+1.701*1.2]
            provenance="current fast camera component centroid ± known1.2m radius; synthetic floating bbox"
        else:
            bbox=data["bboxes"][idx].tolist();circle=agent.hazard._circle_center(bbox)
            provenance="current frozen hazard detector bbox saved in legalcamera fixture"
        row={"step":step,"circle":circle.tolist(),"bbox":bbox,"bbox_provenance":provenance,
            "current_hud_speed_mps":speed,"current_hud_yaw_radps":yaw,
            "previous_emitted_steer":float(trace[step-1]["action"][0]),
            "one_pixel_center_uncertainty_m":uncertainty,"comparisons":{}}
        for kind in ("hermite","quintic"):
            x,y,slope,curve,arc,fit=construction(road,circle,yaw,speed,kind)
            row["comparisons"][kind]={"fit":fit,"geometry":geometry(agent.hazard,road,x,y,slope,curve,circle,uncertainty),
                "dynamics":speed_envelope(arc,curve,speed,row["previous_emitted_steer"]),
                "start_curvature_per_m":float(curve[0]),"curvature_at1p5_m":float(np.interp(1.5,y,curve)),
                "path":{"x":x.tolist(),"y":y.tolist(),"arc":arc.tolist(),"curvature":curve.tolist()}}
        rows.append(row)
    output=Path("agents/apex_2026/results/speed-20261005/quintic_preview_study.json")
    report={"classification":"offline consumed camera58/59 fixed path diagnostic; no episode or controller",
        "camera_helper_source_sha256":source_sha,"fixture_sha256":hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "research_source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "construction":"one fixed left-side4.0m route per camera, same road fit/destination/join/end; compare originalC1 Hermite with C2 quintic matched to HUDyaw/speed atstart",
        "model":{"reaction_s":.08,"lateral_and_total_circle_budget_mps2":190.,"braking_budget_mps2":100.,"joint_rad":.4,"motor_radps":3.,"action_slew_rad":.24,"vehicle_halfwidth_m":1.6,"road_margin_m":.30,"obstacle_radius_m":1.2,"obstacle_margin_m":.75,"minimum_separation_m":3.5},
        "limits":["Illustrative scalar friction circle, not actual four-tire dynamics/reachability or safety certificate.","HUDyaw quantization/bias and camera branch/centroid ambiguity remain.","Single fixed construction and both consumed cameras only; no parameter search/newlap/holdout.","One-pixel sensitivity is an assumed conservative diagnostic bound, not a validated camera-error guarantee."],"rows":rows,"adopted":False,"new_laps":0}
    output.write_text(json.dumps(report,indent=2)+"\n")
    for row in rows:
        print(json.dumps({"step":row["step"],"speed":row["current_hud_speed_mps"],
            "comparisons":{k:{"geometry":v["geometry"],"dynamics":{x:v["dynamics"][x] for x in
            ["reaction_distance_m","reaction_lateral_ok","joint_ok","initial_action_slew_ok","after_reaction_braking_entry_cap_mps","feasible_under_model","first_max_braking_violation_arc_m"]}}
            for k,v in row["comparisons"].items()}}))


if __name__=="__main__":
    main()
