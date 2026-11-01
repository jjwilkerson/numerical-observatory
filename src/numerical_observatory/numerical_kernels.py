"""Optional strict-IEEE CPU kernels. No I/O, domain logic or geodesic solver.

Only imported by an explicitly selected numba backend. No fastmath or parallel
reductions: float64 weights, geometry and corner order follow the reference.
Rejected/ill-conditioned queries are resolved by the original Python evaluator.
"""
import numpy as np
from numba import njit


@njit(cache=True, fastmath=False, error_model='numpy')
def physical(f):
    for v in f:
        if not np.isfinite(v):return False
    if f[0]<=0 or f[4]<=0:return False
    # Same Cholesky criterion, with an ambiguity guard for near-singular gamma.
    scale=max(abs(f[4]),abs(f[7]),abs(f[9]));guard=64*np.finfo(np.float64).eps*scale
    a=np.sqrt(f[4]);b=f[5]/a;c=f[6]/a;d=f[7]-b*b
    if d<=guard:return False
    d=np.sqrt(d);e=(f[8]-c*b)/d;last=f[9]-c*c-e*e
    return last>guard and f[4]>guard


@njit(cache=True, fastmath=False, error_model='numpy')
def candidate(x,p,origins,spacing,shapes):
    for d in range(3):
        if x[d]<origins[p,d] or x[d]>origins[p,d]+shapes[p,d]*spacing[p,d]:return False
    return True


@njit(cache=True, fastmath=False, error_model='numpy')
def gather(arrays,levels,origins,spacing,shapes,level,points):
    values=np.zeros((len(points),10));complete=True
    for n in range(len(points)):
        found=False
        for p in range(len(levels)):
            if levels[p]!=level:continue
            idx=np.empty(3,np.int64);valid=True
            for d in range(3):
                u=(points[n,d]-origins[p,d])/spacing[p,d]
                v=np.rint(u)
                if v<0 or v>=shapes[p,d] or abs(u-v)>=1e-8:valid=False;break
                idx[d]=int(v)
            if not valid:continue
            array=arrays[p]  # Retain the view once, not ten typed-list lookups.
            for f in range(10):
                v=array[idx[2],idx[1],idx[0],f]
                if found and abs(v-values[n,f])>1e-11*(1+abs(values[n,f])):return values,2
                values[n,f]=v
            found=True
        # All overlap consistency checks precede the missing-stencil decision.
        # Otherwise linear fallback could conceal an inconsistent fine patch.
        if not found:complete=False
    return values,0 if complete else 1


@njit(cache=True, fastmath=False, error_model='numpy')
def interpolate(arrays,levels,origins,spacing,shapes,level_ids,mode,p,x,width,reject_finer,stats):
    result=np.zeros((4,10));lo=np.empty(3,np.int64);weights=np.empty((3,2,4))
    for d in range(3):
        u=(x[d]-origins[p,d])/spacing[p,d]
        if not np.isfinite(u) or abs(u)>2**31-5:return result,3
        index=int(np.floor(u))
        if mode!=4 and u<=shapes[p,d]-1:index=min(index,shapes[p,d]-2)
        lo[d]=index;t=u-index;t2=t*t;t3=t2*t
        if width==4:
            weights[d,0,0]=-.5*t+t2-.5*t3;weights[d,0,1]=1-2.5*t2+1.5*t3
            weights[d,0,2]=.5*t+2*t2-1.5*t3;weights[d,0,3]=-.5*t2+.5*t3
            weights[d,1,0]=-.5+2*t-1.5*t2;weights[d,1,1]=-5*t+4.5*t2
            weights[d,1,2]=.5+4*t-4.5*t2;weights[d,1,3]=-t+1.5*t2
        else:
            weights[d,0,0]=1-t;weights[d,0,1]=t;weights[d,1,0]=-1.;weights[d,1,1]=1.
        for j in range(width):weights[d,1,j]/=spacing[p,d]
    if reject_finer:
        for q in range(len(levels)):
            if levels[q]<=levels[p]:continue
            overlap=True
            for d in range(3):
                lower=origins[p,d]+(lo[d]-1)*spacing[p,d];upper=origins[p,d]+(lo[d]+2)*spacing[p,d]
                if lower>origins[q,d]+(shapes[q,d]-1)*spacing[q,d] or upper<origins[q,d]:overlap=False;break
            if overlap:return result,4
    points=np.empty((width**3,3));n=0
    shift=1 if width==4 else 0
    for k in range(width):
        for j in range(width):
            for i in range(width):
                points[n,0]=origins[p,0]+(lo[0]+i-shift)*spacing[p,0]
                points[n,1]=origins[p,1]+(lo[1]+j-shift)*spacing[p,1]
                points[n,2]=origins[p,2]+(lo[2]+k-shift)*spacing[p,2];n+=1
    nodes,error=gather(arrays,levels,origins,spacing,shapes,levels[p],points)
    if error:return result,error
    n=0;coefficients=np.empty((4,width**3))
    for k in range(width):
        for j in range(width):
            for i in range(width):
                for derivative in range(4):
                    coefficients[derivative,n]=weights[0,int(derivative==1),i]*weights[1,int(derivative==2),j]*weights[2,int(derivative==3),k]
                n+=1
    result=coefficients@nodes
    stats[0]+=1
    for l in range(len(level_ids)):
        if level_ids[l]==levels[p]:stats[l+1]=1
    return result,0


@njit(cache=True, fastmath=False, error_model='numpy')
def independent(arrays,levels,origins,spacing,shapes,level_ids,anchors,mode,level,x,stats):
    chosen=-1
    for p in range(len(levels)):
        if levels[p]==level and candidate(x,p,origins,spacing,shapes):chosen=p;break
    if chosen<0:return np.zeros((4,10)),3
    if mode==4:
        for l in range(len(level_ids)):
            if level_ids[l]==level:chosen=anchors[l];break
    result,error=interpolate(arrays,levels,origins,spacing,shapes,level_ids,mode,chosen,x,4,False,stats)
    if not error and not physical(result[0]):error=5
    return result,error


@njit(cache=True, fastmath=False, error_model='numpy')
def decreasing(z,lo,hi):
    u=(z-lo)/(hi-lo);v=1-u
    w=1-u*u*u*(10+u*(-15+6*u)) if u<=.5 else v*v*v*(10+v*(-15+6*v))
    return w,-30*u*u*v*v/(hi-lo)


@njit(cache=True, fastmath=False, error_model='numpy')
def branch(x,b):
    for d in range(3):
        if x[d]<=b[0,d] or x[d]>=b[3,d]:return 0
    for d in range(3):
        if x[d]<b[1,d] or x[d]>b[2,d]:return 2
    return 1


@njit(cache=True, fastmath=False, error_model='numpy')
def box_weight(x,b):
    v=np.ones(3);grad=np.zeros(3)
    for d in range(3):
        if x[d]<=b[0,d] or x[d]>=b[3,d]:v[d]=0
        elif x[d]<b[1,d]:
            w,dw=decreasing(x[d],b[0,d],b[1,d]);v[d]=1-w;grad[d]=-dw
        elif x[d]>b[2,d]:v[d],grad[d]=decreasing(x[d],b[2,d],b[3,d])
    return v[0]*v[1]*v[2],np.array([grad[0]*v[1]*v[2],v[0]*grad[1]*v[2],v[0]*v[1]*grad[2]])


@njit(cache=True, fastmath=False, error_model='numpy')
def blend(coarse,fine,w,dw):
    out=(1-w)*coarse+w*fine
    for d in range(3):
        for f in range(10):out[d+1,f]+=dw[d]*(fine[0,f]-coarse[0,f])
    return out


@njit(cache=True, fastmath=False, error_model='numpy')
def spatial(arrays,levels,origins,spacing,shapes,level_ids,anchors,order,mode,bands,band_levels,plane,x,stats):
    if mode in (0,1):
        for p in order:
            if candidate(x,p,origins,spacing,shapes):
                result,error=interpolate(arrays,levels,origins,spacing,shapes,level_ids,mode,p,x,2 if mode==0 else 4,mode==1,stats)
                if mode==1 or error!=1:return result,error
        return np.zeros((4,10)),3
    if mode==2:
        axis=int(plane[0]);coarse=int(plane[1]);fine=int(plane[2]);lower=plane[3];upper=plane[4];fine_lower=bool(plane[5]);z=x[axis]
        if z<lower:return independent(arrays,levels,origins,spacing,shapes,level_ids,anchors,mode,fine if fine_lower else coarse,x,stats)
        if z>upper:return independent(arrays,levels,origins,spacing,shapes,level_ids,anchors,mode,coarse if fine_lower else fine,x,stats)
        a,error=independent(arrays,levels,origins,spacing,shapes,level_ids,anchors,mode,coarse,x,stats)
        if error:return a,error
        b,error=independent(arrays,levels,origins,spacing,shapes,level_ids,anchors,mode,fine,x,stats)
        if error:return b,error
        w,dw=decreasing(z,lower,upper)
        if not fine_lower:w=1-w;dw=-dw
        gradient=np.zeros(3);gradient[axis]=dw;result=blend(a,b,w,gradient)
        return result,0 if physical(result[0]) else 5
    start=0;base=band_levels[0,0]
    # A fine-core branch prunes all earlier levels, exactly as recursive Python.
    for i in range(len(bands)):
        if branch(x,bands[i])==1:start=i+1;base=band_levels[i,1]
    result,error=independent(arrays,levels,origins,spacing,shapes,level_ids,anchors,mode,base,x,stats)
    if error:return result,error
    for i in range(start,len(bands)):
        if branch(x,bands[i])!=2:continue
        fine,error=independent(arrays,levels,origins,spacing,shapes,level_ids,anchors,mode,band_levels[i,1],x,stats)
        if error:return fine,error
        w,dw=box_weight(x,bands[i]);result=blend(result,fine,w,dw)
        if not physical(result[0]):return result,5
    return result,0


@njit(cache=True, fastmath=False, error_model='numpy')
def adm(left,right,levels,origins,spacing,shapes,level_ids,anchors,order,mode,bands,band_levels,plane,x,w,dt):
    stats=np.zeros(len(level_ids)+1,np.int64)
    a,error=spatial(left,levels,origins,spacing,shapes,level_ids,anchors,order,mode,bands,band_levels,plane,x,stats)
    if error:return a[0].copy(),a.copy(),stats,error
    b,error=spatial(right,levels,origins,spacing,shapes,level_ids,anchors,order,mode,bands,band_levels,plane,x,stats)
    f=(1-w)*a[0]+w*b[0];df=np.empty((4,10));df[0]=(b[0]-a[0])/dt
    df[1:]=(1-w)*a[1:]+w*b[1:]
    if not error and not physical(f):error=5
    return f,df,stats,error


@njit(cache=True, fastmath=False, error_model='numpy')
def reconstruct(f,df):
    gamma=np.array([[f[4],f[5],f[6]],[f[5],f[7],f[8]],[f[6],f[8],f[9]]]);inv=np.linalg.inv(gamma)
    g=np.empty((4,4));dg=np.empty((4,4,4));A=1/(f[0]*f[0]);g[0,0]=-A
    components=np.array([[4,5,6],[5,7,8],[6,8,9]])
    for i in range(3):
        g[0,i+1]=g[i+1,0]=f[i+1]*A
        for j in range(3):g[i+1,j+1]=inv[i,j]-f[i+1]*f[j+1]*A
    for d in range(4):
        dA=-2*A*df[d,0]/f[0];dg[d,0,0]=-dA
        dgamma=np.empty((3,3))
        for i in range(3):
            for j in range(3):dgamma[i,j]=df[d,components[i,j]]
        geometric=-inv@dgamma@inv
        for i in range(3):
            dg[d,0,i+1]=dg[d,i+1,0]=df[d,i+1]*A+f[i+1]*dA
            for j in range(3):
                dg[d,i+1,j+1]=geometric[i,j]-(df[d,i+1]*f[j+1]+f[i+1]*df[d,j+1])*A-f[i+1]*f[j+1]*dA
    return g,dg


@njit(cache=True, fastmath=False, error_model='numpy')
def metric(left,right,levels,origins,spacing,shapes,level_ids,anchors,order,mode,bands,band_levels,plane,x,w,dt):
    f,df,stats,error=adm(left,right,levels,origins,spacing,shapes,level_ids,anchors,order,mode,bands,band_levels,plane,x,w,dt)
    if error:return np.zeros((4,4)),np.zeros((4,4,4)),stats,error
    g,dg=reconstruct(f,df)
    if not np.isfinite(g).all() or not np.isfinite(dg).all():error=6
    return g,dg,stats,error
