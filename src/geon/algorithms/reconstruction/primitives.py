"""Cylinder fitting, OCCT intersection sampling, and conforming UV refinement."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares

@dataclass
class Cylinder:
    origin: np.ndarray
    axis: np.ndarray
    u: np.ndarray
    v: np.ndarray
    radius: float
    theta_start: float
    angular_span: float
    periodic: bool
    rmse: float
    kind = 'cylinder'

    def normal_at(self, uv):
        theta = np.asarray(uv)[..., 0]/self.radius + self.theta_start
        return np.cos(theta)[..., None]*self.u + np.sin(theta)[..., None]*self.v

    def descriptor(self, bounds):
        bounds = np.asarray(bounds, dtype=float).copy()
        bounds[[0, 2]] = bounds[[0, 2]]/self.radius + self.theta_start
        # A complete periodic surface has exactly one angular revolution.
        if bounds[2]-bounds[0] > 2*np.pi:
            bounds[0], bounds[2] = self.theta_start, self.theta_start+2*np.pi
        return dict(kind=self.kind, origin=self.origin, normal=self.axis,
                    u=self.u, radius=self.radius, bounds=bounds)

    def project(self, xyz):
        q=np.asarray(xyz)-self.origin
        theta=(np.arctan2(q@self.v,q@self.u)-self.theta_start)%(2*np.pi)
        # Preserve leading dimensions, including a single point (3,) -> (2,).
        # Corner reconciliation mixes normals from all surface types.
        return np.stack([self.radius*theta,q@self.axis],axis=-1)

    def lift(self, uv):
        uv=np.asarray(uv);theta=uv[...,0]/self.radius+self.theta_start
        return self.origin+self.radius*(np.cos(theta)[...,None]*self.u+np.sin(theta)[...,None]*self.v)+uv[...,1,None]*self.axis

    def residual(self, xyz):
        q=np.asarray(xyz)-self.origin
        return np.linalg.norm(q-(q@self.axis)[...,None]*self.axis,axis=-1)-self.radius


def fit_cylinder(points,max_radius=10.,max_samples=2048):
    xyz=np.asarray(points);origin=xyz.mean(axis=0);q=xyz-origin
    sample=q[np.random.default_rng(0).choice(len(q),min(len(q),max_samples),replace=False)]
    _,basis=np.linalg.eigh(sample.T@sample/len(sample))
    extent=max(np.ptp(sample,axis=0).max(),.05)
    candidates=[]
    for k in (2,0,1):
        axis0=basis[:,k];u=basis[:,(k+1)%3];v=np.cross(axis0,u)
        x,y=sample@u,sample@v
        coef=np.linalg.lstsq(np.column_stack([2*x,2*y,np.ones(len(x))]),x*x+y*y,rcond=None)[0]
        radius=np.sqrt(max(1e-6,coef[2]+coef[0]**2+coef[1]**2))
        if not .01<radius<max_radius:continue
        def model(z):
            axis=axis0+z[2]*u+z[3]*v;axis/=np.linalg.norm(axis)
            center=z[0]*u+z[1]*v
            return center,axis,np.exp(z[4])
        def residual(z):
            center,axis,r=model(z);d=sample-center
            return np.linalg.norm(d-(d@axis)[:,None]*axis,axis=1)-r
        z0=[coef[0],coef[1],0,0,np.log(radius)]
        bound=max(extent*5,max_radius*2)
        try:
            fit=least_squares(residual,z0,bounds=([-bound,-bound,-3,-3,np.log(.005)],[bound,bound,3,3,np.log(max_radius)]),loss='soft_l1',f_scale=.01,max_nfev=90)
        except ValueError:continue
        c,axis,r=model(fit.x);center=origin+c
        u1=u-axis*(u@axis);u1/=np.linalg.norm(u1);v1=np.cross(axis,u1)
        d=xyz-center;radial=np.linalg.norm(d-(d@axis)[:,None]*axis,axis=1)
        rmse=np.sqrt(np.mean((radial-r)**2))
        angles=np.sort(np.arctan2(d@v1,d@u1)%(2*np.pi));gaps=np.diff(np.r_[angles,angles[0]+2*np.pi]);gap=int(np.argmax(gaps))
        start=float(angles[(gap+1)%len(angles)])-1e-9
        span=float(2*np.pi-gaps[gap]);periodic=gaps[gap]<np.deg2rad(15)
        candidates.append(Cylinder(center,axis,u1,v1,float(r),start,span,periodic,float(rmse)))
    return min(candidates,key=lambda c:c.rmse) if candidates else None


def curve_in_patch(patch, xyz):
    uv=patch.project(xyz)
    if getattr(patch, "cylinder", None) is not None and len(uv)>1:
        r=patch.cylinder.radius;uv[:,0]=np.unwrap(uv[:,0]/r)*r
        period=2*np.pi*r
        if patch.footprint is not None:
            middle=(patch.footprint.bounds[0]+patch.footprint.bounds[2])/2
            uv[:,0]+=np.round((middle-np.mean(uv[:,0]))/period)*period
    return uv


def primitive_residual(patch, xyz):
    from .geometry_backend import surface_of
    return surface_of(patch).residual(xyz)


def raw_intersection_curves(a, b, step, chord_tolerance=.005, margin=0., diagnostic=None):
    """OCCT intersections for every supported surface pair."""
    from .geometry_backend import intersection_curves
    return intersection_curves(a, b, step, chord_tolerance, margin, diagnostic)


def simplify_curve_3d(points,tolerance):
    if len(points)<3 or tolerance<=0:return points
    keep={0,len(points)-1};stack=[(0,len(points)-1)]
    while stack:
        a,b=stack.pop();v=points[b]-points[a];length=v@v
        t=np.clip((points[a+1:b]-points[a])@v/max(length,1e-20),0,1)
        dist=np.linalg.norm(points[a+1:b]-(points[a]+t[:,None]*v),axis=1)
        if len(dist) and dist.max()>tolerance:
            k=a+1+int(np.argmax(dist));keep.add(k);stack.extend([(a,k),(k,b)])
    return points[sorted(keep)]


def refine_surface_triangles(polygons, lift, tolerance, locked_curves=(), spacing=None):
    """Conforming two-direction refinement, preserving common seam segments."""
    import shapely
    vertices, lookup, faces = [], {}, []
    def index(p):
        key = tuple(np.round(p, 11))
        if key not in lookup:
            lookup[key] = len(vertices)
            vertices.append(np.asarray(p))
        return lookup[key]
    if spacing is None:
        initial=[np.asarray(t.exterior.coords)[:3] for p in polygons
                 for t in shapely.constrained_delaunay_triangles(p).geoms]
    else:
        from .meshing import grid_triangles
        initial=grid_triangles(polygons,spacing)
    for triangle in initial:
        faces.append([index(p) for p in triangle])
    locked = set()
    for curve in locked_curves:
        for a, b in zip(curve[:-1], curve[1:]):
            locked.add(tuple(sorted((index(a), index(b)))))
    for iteration in range(24):
        if len(faces) > 1000000:
            raise ValueError('Spline triangulation exceeded one million triangles')
        edges = sorted({tuple(sorted((a, b))) for tri in faces for a, b in zip(tri, np.roll(tri, -1))})
        uv = np.asarray(vertices)
        xyz = lift(uv)
        split = {}
        for start in range(0, len(edges), 65536):
            block = np.asarray(edges[start:start+65536], dtype=int)
            if not len(block):
                continue
            midpoint = uv[block].mean(axis=1)
            error = np.linalg.norm(lift(midpoint)-xyz[block].mean(axis=1), axis=1)
            for edge, mid in zip(block[error > tolerance], midpoint[error > tolerance]):
                key = tuple(edge)
                if key not in locked:
                    split[key] = index(mid)
        refined, interior_split = [], False
        for tri in faces:
            mids = [split.get(tuple(sorted((tri[i], tri[(i+1)%3])))) for i in range(3)]
            count = sum(m is not None for m in mids)
            if count == 0:
                center = uv[tri].mean(axis=0)
                if np.linalg.norm(lift(center)-xyz[tri].mean(axis=0)) > tolerance:
                    m = index(center)
                    refined.extend([[tri[i], tri[(i+1)%3], m] for i in range(3)])
                    interior_split = True
                else:
                    refined.append(tri)
            elif count == 3:
                a,b,c = tri; x,y,z = mids
                refined.extend([[a,x,z], [x,b,y], [z,y,c], [x,y,z]])
            else:
                i = (next(i for i,m in enumerate(mids) if m is not None) if count == 1 else
                     next(i for i in range(3) if mids[i] is not None and mids[(i+1)%3] is not None))
                a,b,c = [tri[(i+j)%3] for j in range(3)]; x = mids[i]
                if count == 1:
                    refined.extend([[a,x,c], [x,b,c]])
                else:
                    y=mids[(i+1)%3];refined.extend([[a,x,c], [x,y,c], [x,b,y]])
        faces = refined
        if not split and not interior_split:
            return np.asarray(vertices)[np.asarray(faces, dtype=int)] if faces else np.empty((0,3,2))
    raise ValueError('Spline triangulation did not converge to chord tolerance')


def refine_uv_triangles(polygons, max_dx):
    """Conforming refinement; shared UV edges use the same midpoint index."""
    import shapely
    from shapely.geometry import Polygon
    parts=[]
    for poly in polygons:
        rings=[]
        for ring in [poly.exterior,*poly.interiors]:
            uv=np.asarray(ring.coords)[:-1];densified=[]
            for a,b in zip(uv,np.roll(uv,-1,axis=0)):
                count=max(1,int(np.ceil(abs(b[0]-a[0])/max_dx)))
                densified.extend(a+np.arange(count)[:,None]/count*(b-a))
            rings.append(np.asarray(densified))
        parts.extend(shapely.constrained_delaunay_triangles(Polygon(rings[0],rings[1:])).geoms)
    vertices=[];lookup={};faces=[]
    def index(p):
        key=tuple(np.round(p,11))
        if key not in lookup:lookup[key]=len(vertices);vertices.append(np.asarray(p))
        return lookup[key]
    for triangle in parts:faces.append([index(p) for p in np.asarray(triangle.exterior.coords)[:3]])
    for _ in range(24):
        long={}
        for tri in faces:
            for a,b in zip(tri,np.roll(tri,-1)):
                key=tuple(sorted((a,b)))
                if abs(vertices[a][0]-vertices[b][0])>max_dx*(1+1e-8):long[key]=None
        if not long:break
        for key in long:long[key]=index((vertices[key[0]]+vertices[key[1]])/2)
        refined=[]
        for tri in faces:
            mids=[long.get(tuple(sorted((tri[i],tri[(i+1)%3])))) for i in range(3)]
            count=sum(m is not None for m in mids)
            if count==0:refined.append(tri);continue
            if count==3:
                a,b,c=tri;x,y,z=mids;refined.extend([[a,x,z],[x,b,y],[z,y,c],[x,y,z]]);continue
            # Rotate so a split edge is a-b; for two edges choose a-b and b-c.
            if count==1:i=next(i for i,m in enumerate(mids) if m is not None)
            else:i=next(i for i in range(3) if mids[i] is not None and mids[(i+1)%3] is not None)
            a,b,c=[tri[(i+j)%3] for j in range(3)];x=mids[i]
            if count==1:refined.extend([[a,x,c],[x,b,c]])
            else:
                y=mids[(i+1)%3];refined.extend([[a,x,c],[x,y,c],[x,b,y]])
        faces=refined
    return np.asarray(vertices)[np.asarray(faces,dtype=int)] if faces else np.empty((0,3,2))
