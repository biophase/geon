"""Disposable spline fitting overlays; no control geometry is persisted in meshes."""
import numpy as np
import shapely
from shapely.geometry import LineString

from .meshing import spline_grid_spacing


def spline_preview(result, tolerance=.005, progress=None):
    overlays={}
    for label,patch in result.patches.items():
        if patch.kind!='bspline':
            continue
        if progress:
            progress(f'Preparing B-spline UV grid and control net: primitive {label}')
        model=patch.spline
        bounds=np.asarray(patch.polygon.bounds)
        step=spline_grid_spacing(model,.5,tolerance)
        grid=[]
        for axis in range(2):
            for value in np.linspace(bounds[axis],bounds[axis+2],11):
                ends=np.array([bounds[:2],bounds[2:]])
                ends[:,axis]=value
                clipped=shapely.intersection(LineString(ends),patch.polygon)
                for part in shapely.get_parts(clipped):
                    if part.geom_type!='LineString' or part.length<=1e-10:
                        continue
                    uv=np.asarray(part.coords)
                    for a,b in zip(uv[:-1],uv[1:]):
                        n=min(4096,max(2,int(np.ceil(np.linalg.norm(b-a)/step))+1))
                        grid.append(model.lift(a+np.linspace(0,1,n)[:,None]*(b-a)))
        poles=model.descriptor()['poles']
        net=[row for row in poles]+[col for col in poles.transpose(1,0,2)]
        # Explicit dashed geometry also works on renderers without line stipple.
        period=max(float(np.min(model.bounds[2:]-model.bounds[:2]))/8,1e-5)
        dashed=[]
        for line in net:
            for a,b in zip(line[:-1],line[1:]):
                length=np.linalg.norm(b-a)
                n=min(512,max(2,int(np.ceil(length/period))))
                for i in range(n):
                    dashed.append(np.array([a+(b-a)*i/n,a+(b-a)*(i+.5)/n]))
        overlays[label]=dict(grid=grid,control_net=dashed,control_points=poles.reshape(-1,3))
    return overlays
