"""Camera-only bounded normal-offset quadratic prototype, not a candidate."""
import numpy as np


def normal_line(path, radius=3., iterations=800):
    # Uniform arc sampling avoids row-coordinate hairpin singularities.
    s = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
    at=np.arange(0.,s[-1]+.01,2.)
    center=np.column_stack([np.interp(at,s,path[:,j]) for j in (0,1)])
    tangent=np.gradient(center,axis=0)
    tangent/=np.maximum(np.linalg.norm(tangent,axis=1)[:,None],.001)
    normal=np.column_stack((tangent[:,1],-tangent[:,0]))
    # Ego position and heading are physical camera coordinates, not road center.
    prefix=np.array([[0.,-1.],[0.,0.]])
    n=len(center);d2=np.diff(np.eye(n+2),n=2,axis=0)
    base=np.vstack((prefix,center));nx=np.r_[0.,0.,normal[:,0]];ny=np.r_[0.,0.,normal[:,1]]
    bx=d2[:,2:]*nx[None,2:];by=d2[:,2:]*ny[None,2:]
    h=bx.T@bx+by.T@by+.005*np.eye(n)
    linear=bx.T@(d2@base[:,0])+by.T@(d2@base[:,1])
    step=1./np.max(np.sum(abs(h),axis=1))
    x=np.zeros(n);z=x.copy();momentum=1.
    for _ in range(iterations):
        new=np.clip(z-step*(h@z+linear),-radius,radius)
        m=.5*(1.+np.sqrt(1.+4.*momentum*momentum))
        z=new+(momentum-1.)/m*(new-x);x=new;momentum=m
    return center+x[:,None]*normal


def footprint_depth(agent, field, path):
    delta=np.diff(path,axis=0);length=np.linalg.norm(delta,axis=1)
    positions=[];directions=[]
    for a,b,t,l in zip(path[:-1],path[1:],delta,length):
        phase=np.linspace(0.,1.,max(2,int(np.ceil(l/.5))+1))
        positions.append(a+phase[:,None]*(b-a));directions.append(np.tile(t/max(l,.001),(len(phase),1)))
    p=np.concatenate(positions);t=np.concatenate(directions)
    n=np.column_stack((t[:,1],-t[:,0]))
    corners=np.concatenate([p+lat*n+long*t for lat in (-1.6,1.6) for long in (-2.4,2.6)])
    return float(np.min(agent._sample_distance(field,corners)))
