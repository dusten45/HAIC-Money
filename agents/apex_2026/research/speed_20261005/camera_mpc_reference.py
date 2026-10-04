"""NumPy-only camera-reference geometry for predictive-control research.

No agent, privileged labels or simulator lookup. Pose angle is conventional
CCW: local forward rotates to[-sin(angle),cos(angle)] in camera coordinates.
Distance fields are supplied by camera perception; outside is unsupported.
"""
import numpy as np


def arclength(path):
    return np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]


def project_one(path,point,lower,upper):
    """Nearest projection restricted to an ordered reachable arc window."""
    arc=arclength(path);links=np.diff(path,axis=0);length=np.diff(arc)
    active=(arc[:-1]<=upper)&(arc[1:]>=lower)&(length>1e-8)
    indices=np.flatnonzero(active)
    if not len(indices):raise ValueError('Projection interval misses supported reference')
    lo=np.clip((lower-arc[indices])/length[indices],0.,1.)
    hi=np.clip((upper-arc[indices])/length[indices],0.,1.)
    fraction=np.sum((point-path[indices])*links[indices],axis=1)/length[indices]**2
    fraction=np.clip(fraction,lo,hi)
    projected=path[indices]+fraction[:,None]*links[indices]
    chosen=int(np.argmin(np.sum((projected-point)**2,axis=1)))
    index=indices[chosen];tangent=links[index]/length[index]
    lateral=float(np.dot(point-projected[chosen],[tangent[1],-tangent[0]]))
    return float(arc[index]+fraction[chosen]*length[index]),lateral,tangent


def project_progress(path,positions,velocity,dt,projection_uncertainty_m=0.):
    """Progress along the reference, without a road-center rejoin penalty.

    A backward first-tangent extension establishes ego arc origin only; it
    does not invent supported asphalt. Future projection is monotone and
    cannot advance more than max(prior,post speed)*dt plus supplied
    uncertainty, respecting semi-implicit post-velocity pose integration.
    """
    path=np.asarray(path,float);positions=np.asarray(positions,float)
    velocity=np.asarray(velocity,float)
    direction=path[1]-path[0];direction/=np.linalg.norm(direction)
    extended=np.vstack((path[0]-10.*direction,path))
    origin,_,_=project_one(extended,positions[0],0.,10.)
    current=origin;rows=[];extent=arclength(extended)[-1]
    for i,point in enumerate(positions):
        advance=0. if i==0 else float(max(np.linalg.norm(velocity[i-1]),np.linalg.norm(velocity[i]))*dt+projection_uncertainty_m)
        current,lateral,tangent=project_one(extended,point,current,min(extent,current+advance))
        rows.append((current-origin,lateral,tangent[0],tangent[1]))
    return dict(projections=np.asarray(rows),supported_remaining_m=float(extent-current),
                reference_origin_arc_m=float(origin-10.),supported_extent_from_ego_m=float(extent-origin))


def body_points(positions,angles):
    """Full25-point rectangle stencil, including edges/interior and corners."""
    positions=np.asarray(positions,float);angles=np.asarray(angles,float)
    lateral,longitudinal=np.meshgrid(np.linspace(-1.6,1.6,5),np.linspace(-2.4,2.6,5))
    lat=lateral.ravel();long=longitudinal.ravel();c=np.cos(angles)[:,None];s=np.sin(angles)[:,None]
    offsets=np.stack((c*lat-s*long,s*lat+c*long),axis=2)
    return positions[:,None,:]+offsets


def sample_field(field,points):
    """Bilinear camera field sampling; outside road image rows0..72 is0."""
    shape=points.shape[:-1];points=np.asarray(points).reshape(-1,2)
    x=42.+1.3608*points[:,0];y=63.-1.701*points[:,1]
    valid=(x>=0.)&(x<=83.)&(y>=0.)&(y<=72.)
    x=np.clip(x,0,83);y=np.clip(y,0,72);ix=x.astype(int);iy=y.astype(int)
    nx=np.minimum(ix+1,83);ny=np.minimum(iy+1,72);wx=x-ix;wy=y-iy
    values=(1-wx)*(1-wy)*field[iy,ix]+wx*(1-wy)*field[iy,nx]+(1-wx)*wy*field[ny,ix]+wx*wy*field[ny,nx]
    return np.where(valid,values,0.).reshape(shape)


def road_slack(field,positions,angles,minimum_depth_m=1.9,grass_mask=None):
    points=body_points(positions,angles)
    depth=sample_field(field,points)
    if grass_mask is not None:
        # Caller must distinguish camera grass from painted/occluded asphalt.
        grass=sample_field(np.asarray(grass_mask,float),points)
        depth=np.where(grass>.5,0.,depth)
    return np.min(depth,axis=1)-minimum_depth_m


def obstacle_slack(positions,angles,circles,uncertainty_m=0.):
    """Detected/transported camera centers are(x,y), unlike detector(y,x)."""
    positions=np.asarray(positions);angles=np.asarray(angles)
    if not len(circles):return np.full(len(positions),np.inf)
    centers=np.asarray(circles,float);delta=centers[None]-positions[:,None,:]
    center=np.linalg.norm(delta,axis=2)-(3.7+uncertainty_m)
    c=np.cos(angles)[:,None];s=np.sin(angles)[:,None]
    x=c*delta[:,:,0]+s*delta[:,:,1];y=-s*delta[:,:,0]+c*delta[:,:,1]
    dx=x-np.clip(x,-1.6,1.6);dy=y-np.clip(y,-2.4,2.6)
    rectangle=np.hypot(dx,dy)-(1.2+.75+uncertainty_m)
    return np.min(np.minimum(center,rectangle),axis=1)


def terminal_speed_limit(reference,angle,tangent,speed,braking_accel,braking_delay_s,uncertainty_m=0.):
    """Stop before supported reference end; supplied braking model is assumed."""
    if braking_accel<=0. or braking_delay_s<0.:raise ValueError('Invalid braking assumption')
    offsets=body_points(np.zeros((1,2)),np.asarray([angle]))[0]
    front=max(0.,float(np.max(offsets@np.asarray(tangent))))
    usable=max(0.,reference['supported_remaining_m']-front-speed*braking_delay_s-uncertainty_m)
    return float(np.sqrt(2.*braking_accel*usable))


class CameraGeometry:
    """Frozen reference and camera support for one MPC planning call.

    evaluate inputs include the initial state before raw integration ticks.
    Re-evaluate each full beam prefix so branch progress survives .08 knots.
    circles are detected/remembered camera(x,y), never detector(y,x).
    """
    def __init__(self,path,field,circles=(),grass_mask=None,
                 minimum_depth_m=1.9,obstacle_uncertainty_m=0.,projection_uncertainty_m=0.):
        self.path=np.asarray(path,float).copy();self.field=np.asarray(field,float).copy()
        self.circles=np.asarray(circles,float).reshape(-1,2).copy()
        self.grass_mask=None if grass_mask is None else np.asarray(grass_mask,bool).copy()
        self.minimum_depth_m=minimum_depth_m
        self.obstacle_uncertainty_m=obstacle_uncertainty_m
        self.projection_uncertainty_m=projection_uncertainty_m
        if len(self.path)<2 or not np.isfinite(self.path).all() or not np.isfinite(self.field).all():
            raise ValueError('Unsupported camera reference')

    def evaluate(self,positions,angles,velocities,dt=.02):
        positions=np.asarray(positions,float);angles=np.asarray(angles,float);velocities=np.asarray(velocities,float)
        if positions.shape!=velocities.shape or positions.shape!=(len(angles),2):
            raise ValueError('Pose/velocity arrays must align')
        if not np.isfinite(positions).all() or not np.isfinite(angles).all() or not np.isfinite(velocities).all():
            raise ValueError('Nonfinite predicted state')
        projection=project_progress(self.path,positions,velocities,dt,self.projection_uncertainty_m)
        road=road_slack(self.field,positions,angles,self.minimum_depth_m,self.grass_mask)
        obstacle=obstacle_slack(positions,angles,self.circles,self.obstacle_uncertainty_m)
        tangents=projection['projections'][:,2:4]
        offsets=body_points(positions,angles)-positions[:,None,:]
        front=np.maximum(0.,np.max(np.sum(offsets*tangents[:,None,:],axis=2),axis=1))
        progress=projection['projections'][:,0]
        front_slack=projection['supported_extent_from_ego_m']-progress-front
        return dict(progress=progress,progress_m=progress,
                    lateral_offset=projection['projections'][:,1],
                    path_tangents=tangents,
                    road_slack=road,road_slack_m=road,
                    obstacle_slack=obstacle,obstacle_slack_m=obstacle,
                    reference_front_slack_m=front_slack,
                    violation=(road<0.)|(obstacle<0.)|(front_slack<0.),
                    supported_remaining_m=projection['supported_remaining_m'],
                    supported_extent_from_ego_m=projection['supported_extent_from_ego_m'])
