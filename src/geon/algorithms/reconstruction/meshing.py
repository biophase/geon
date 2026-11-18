"""Regular UV cells and synchronized seam samples for curved surface meshes."""
import numpy as np
import shapely


def insert_boundary_nodes(geometry, curves, tolerance=1e-8):
    """Subdivide accepted rings in place, without choosing/replacing border chains.

    All new nodes must lie on the existing boundary within numerical tolerance.
    Ring ordering, holes and disconnected components remain unchanged. This is
    deliberately independent of the much larger border-movement tolerance.
    """
    polygons=list(shapely.get_parts(geometry))
    rings=[np.asarray(r.coords)[:-1] for p in polygons for r in (p.exterior,*p.interiors)]
    segments=np.concatenate([np.stack((r,np.roll(r,-1,axis=0)),axis=1) for r in rings])
    points=np.unique(np.concatenate(curves),axis=0)
    tree=shapely.STRtree(shapely.linestrings(segments))
    nodes,edges=tree.query(shapely.points(points),predicate='dwithin',distance=tolerance)
    if len(np.unique(nodes))!=len(points):
        raise ValueError('Shared seam node is not on the accepted patch boundary')
    inserts={}
    for node,edge in zip(nodes,edges):
        a,b=segments[edge]
        delta=b-a
        length=np.linalg.norm(delta)
        if length<=tolerance:
            continue
        t=np.dot(points[node]-a,delta)/(length*length)
        if tolerance/length<t<1-tolerance/length:
            inserts.setdefault(int(edge),[]).append((float(t),points[node]))
    rebuilt=[]
    offset=0
    for ring in rings:
        output=[]
        for i,a in enumerate(ring):
            output.append(a)
            for _,p in sorted(inserts.get(offset+i,[]),key=lambda item:item[0]):
                if np.linalg.norm(p-output[-1])>tolerance:
                    output.append(p)
        rebuilt.append(np.asarray(output))
        offset+=len(ring)
    result=[]
    index=0
    for polygon in polygons:
        count=1+len(polygon.interiors)
        result.append(shapely.Polygon(rebuilt[index],rebuilt[index+1:index+count]))
        index+=count
    result=result[0] if len(result)==1 else shapely.MultiPolygon(result)
    if not result.is_valid:
        raise ValueError('Inserting shared seam nodes produced an invalid boundary')
    return result


def spline_grid_spacing(surface, maximum, tolerance):
    """Bound the height interpolation error over a square UV cell by M*h*h.

    B-spline derivative control values bound the Hessian throughout the domain.
    The chart has physical distance units, so elongated patches get many cells
    along their length instead of a fixed number of cells in normalized UV.
    """
    nu,nv = surface.heights.shape
    u1 = surface._u_basis.derivative().c[:nu-1]
    u2 = surface._u_basis.derivative(2).c[:nu-2]
    v1 = surface._v_basis.derivative().c[:nv-1]
    v2 = surface._v_basis.derivative(2).c[:nv-2]
    width = surface.bounds[2:]-surface.bounds[:2]
    uu = np.abs(u2@surface.heights).max()/width[0]**2
    uv = np.abs(u1@surface.heights@v1.T).max()/np.prod(width)
    vv = np.abs(surface.heights@v2.T).max()/width[1]**2
    bound = max(uu+uv, vv+uv)
    return min(maximum, np.sqrt(tolerance/bound)) if bound>0 else maximum


def grid_crossings(uv, spacing):
    """Return per-segment fractions where a polyline crosses global UV grid lines."""
    spacing=np.broadcast_to(spacing,(2,))
    result=[]
    for a,b in zip(uv[:-1],uv[1:]):
        fractions=[0.]
        for axis in range(2):
            delta=b[axis]-a[axis]
            if abs(delta)<1e-14:
                continue
            low,high=sorted((a[axis],b[axis]))
            step=spacing[axis]
            lines=np.arange(np.floor(low/step)+1,np.ceil(high/step))*step
            t=(lines-a[axis])/delta
            fractions.extend(t[(t>1e-10)&(t<1-1e-10)])
        result.append(np.unique(fractions))
    return result


def synchronize_seam_grid(edge, patches, spacings):
    """Insert the union of both charts' grid crossings into one shared 3D seam."""
    points=edge['points'] if 'points' in edge else edge['endpoints']
    from .primitives import curve_in_patch
    charts={label:curve_in_patch(patches[label],edge.get('side_points',{}).get(label,points))
            for label in edge['patch_ids']}
    fractions=[np.array([0.]) for _ in range(len(points)-1)]
    for label in edge['patch_ids']:
        if label not in spacings:
            continue
        crossings=grid_crossings(charts[label],spacings[label])
        fractions=[np.unique(np.r_[old,new]) for old,new in zip(fractions,crossings)]
    # The two charts can hit the same crossing with slightly different floats.
    # Keep one sample so welding cannot collapse a microscopic seam segment.
    fractions=[t[np.r_[True,np.diff(t)>1e-9] & (t<1-1e-9)] for t in fractions]
    def insert(curve):
        parts=[a+t[:,None]*(b-a) for a,b,t in zip(curve[:-1],curve[1:],fractions)]
        return np.vstack([*parts,curve[-1:]])
    edge['points']=insert(points)
    if 'side_points' not in edge and any(patches[i].kind=='cylinder' for i in edge['patch_ids']):
        # A cylinder chord lies inside the cylinder. Refine newly inserted
        # analytic seam nodes onto both surfaces; leave accepted corners intact.
        mask=np.r_[np.concatenate([t>0 for t in fractions]),False]
        refined=edge['points'][mask].copy()
        if len(refined):
            surfaces=[patches[i].surface for i in edge['patch_ids']]
            for _ in range(12):
                error=np.stack([s.residual(refined) for s in surfaces],axis=-1)
                if np.max(np.abs(error))<1e-9:
                    break
                normals=np.stack([s.normal_at(s.project(refined)) for s in surfaces],axis=1)
                gram=normals@normals.transpose(0,2,1)
                weights=(np.linalg.pinv(gram)@error[...,None])[...,0]
                refined-=np.einsum('ni,nij->nj',weights,normals)
            if not np.isfinite(refined).all() or max(np.max(np.abs(s.residual(refined))) for s in surfaces)>1e-7:
                raise ValueError('Could not refine cylinder seam grid nodes onto both surfaces')
            edge['points'][mask]=refined
    edge['endpoints']=edge['points'][[0,-1]].copy()
    if 'side_points' in edge:
        edge['side_points']={label:insert(curve) for label,curve in edge['side_points'].items()}
    # Cylinder charts are nonlinear: projecting a 3D chord midpoint does not
    # yield the midpoint of its accepted UV segment. Retain paired chart nodes.
    edge['mesh_uv']={label:insert(curve) for label,curve in charts.items()}


def cylinder_grid_spacing(cylinder, maximum, tolerance):
    """Physical arc-length/axial cell widths, bounded by circular chord error."""
    radius=cylinder.radius
    arc=min(maximum,radius*min(np.pi/4,2*np.arccos(np.clip(1-tolerance/radius,-1,1))))
    if cylinder.periodic:
        period=2*np.pi*radius
        arc=period/max(1,int(np.ceil(period/arc)))
    return np.array([arc,min(maximum,arc)])


def grid_triangles(polygons, spacing):
    """Clip a regular grid to the observed polygon, retaining holes and vertices.

    Only boundary cells need constrained triangulation. Shared seam nodes must
    already contain grid crossings from both patches before calling this.
    """
    spacing=np.broadcast_to(spacing,(2,))
    triangles=[]
    for polygon in polygons:
        bounds=np.asarray(polygon.bounds)
        low=np.floor(bounds[:2]/spacing).astype(np.int64)
        high=np.ceil(bounds[2:]/spacing).astype(np.int64)
        shape=np.maximum(high-low,1)
        count=int(np.prod(shape))
        if count>500000:
            raise ValueError('Curved-surface UV grid exceeds 500,000 cells; increase mesh spacing or chord tolerance')
        for start in range(0,count,4096):
            indices=np.arange(start,min(count,start+4096))
            x=(low[0]+indices%shape[0])*spacing[0]
            y=(low[1]+indices//shape[0])*spacing[1]
            cells=shapely.box(x,y,x+spacing[0],y+spacing[1])
            clipped=shapely.intersection(cells,polygon)
            clipped=clipped[shapely.area(clipped)>1e-18]
            parts=shapely.get_parts(shapely.constrained_delaunay_triangles(clipped))
            triangles.extend(np.asarray(p.exterior.coords)[:3] for p in parts)
            if len(triangles)>1000000:
                raise ValueError('Curved-surface mesh exceeds one million triangles')
    return triangles
