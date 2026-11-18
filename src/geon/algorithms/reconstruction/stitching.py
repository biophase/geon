"""Conforming stitching of nearby exposed planar mesh-border intervals."""
import numpy as np
from shapely.geometry import Polygon
import shapely
from scipy.spatial import cKDTree


def stitch_boundaries(vertices,faces,labels,parts,flags,patches,tolerance=.05,min_length=.1,weld=1e-7,cfg=None):
    if tolerance<=0 or not faces:return vertices,faces,labels,parts,flags,[],0
    xyz=np.asarray(vertices);tri=np.asarray(faces);ids=np.asarray(labels)
    edges=np.sort(np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]),axis=1)
    unique,first,count=np.unique(edges,axis=0,return_index=True,return_counts=True)
    source=np.tile(np.arange(len(tri)),3)[first]
    allowed=(count==1)&~np.asarray(flags)[source]
    allowed &= np.array([patches[int(ids[i])].kind in ('plane', 'bspline') for i in source])
    edges=unique[allowed];source=source[allowed];edgeids=ids[source]
    if not len(edges):return vertices,faces,labels,parts,flags,[],0
    a,b=xyz[edges[:,0]],xyz[edges[:,1]];length=np.linalg.norm(b-a,axis=1)
    direction=(b-a)/np.maximum(length[:,None],1e-15)
    third=np.array([next(int(v) for v in tri[f] if v not in e) for e,f in zip(edges,source)],dtype=int)
    inward=xyz[third]-(a+b)/2;inward-=np.einsum('ij,ij->i',inward,direction)[:,None]*direction
    inward/=np.maximum(np.linalg.norm(inward,axis=1)[:,None],1e-15)
    proposals=[]
    for i in range(len(edges)):
        dot=direction@direction[i]
        lo=np.minimum((a-a[i])@direction[i],(b-a[i])@direction[i]);hi=np.maximum((a-a[i])@direction[i],(b-a[i])@direction[i])
        start=np.maximum(0,lo);stop=np.minimum(length[i],hi)
        possible=(np.arange(len(edges))>i)&(edgeids!=edgeids[i])&(np.abs(dot)>.97)&(stop-start>=min_length)
        for j in np.flatnonzero(possible):
            pa,pb=patches[int(edgeids[i])],patches[int(edgeids[j])]
            if pa.part_id!=pb.part_id:continue
            nearly_parallel = abs(pa.normal_at(pa.project(a[i]))@pb.normal_at(pb.project(a[j])))>=np.cos(np.deg2rad(cfg.coplanar_angle_degrees if cfg else 5))
            if cfg is not None:
                if nearly_parallel:
                    if not cfg.join_coplanar:continue
                    if max(abs(pb.residual(a[i])),abs(pa.residual(a[j])))>cfg.coplanar_plane_tolerance:continue
                elif not cfg.bridge_remaining_gaps:continue
            aa=np.array([a[i]+start[j]*direction[i],a[i]+stop[j]*direction[i]])
            t=((aa-a[j])@direction[i])/dot[j]
            bb=a[j]+t[:,None]*direction[j]
            distance=np.linalg.norm(aa-bb,axis=1)
            limit = min(tolerance,cfg.gap_bridge_tolerance) if cfg is not None and not nearly_parallel else tolerance
            if distance.max()>limit:continue
            gap=(bb-aa).mean(axis=0)
            if gap@inward[i]>1e-5 or (-gap)@inward[j]>1e-5:continue
            proposals.append((-(stop[j]-start[j]),float(distance.max()),i,j,aa,bb,
                              (start[j]/length[i],stop[j]/length[i]),tuple(sorted(t/length[j]))))
    occupied={};selected=[]
    for proposal in sorted(proposals,key=lambda x:(x[0],x[1],x[2],x[3])):
        _,_,i,j,aa,bb,ia,ib=proposal
        if any(max(interval[0],previous[0])<min(interval[1],previous[1])-1e-8
               for edge,interval in ((i,ia),(j,ib)) for previous in occupied.get(edge,[])):
            continue
        occupied.setdefault(i,[]).append(ia);occupied.setdefault(j,[]).append(ib);selected.append(proposal)
    if not selected:return vertices,faces,labels,parts,flags,[],0
    newvertices=[p.copy() for p in xyz];lookup={}
    for i,p in enumerate(newvertices):lookup.setdefault(tuple(np.round(p/weld).astype(np.int64)),[]).append(i)
    def index(point):
        cell=np.round(point/weld).astype(np.int64)
        from itertools import product
        for offset in product((-1,0,1),repeat=3):
            for i in lookup.get(tuple(cell+offset),[]):
                if np.linalg.norm(newvertices[i]-point)<=weld:return i
        i=len(newvertices);newvertices.append(point.copy());lookup.setdefault(tuple(cell),[]).append(i);return i
    splits={};records=[];strips=[]
    for _,_,i,j,aa,bb,_,_ in selected:
        for e,curve in ((i,aa),(j,bb)):
            key=tuple(edges[e]);values=splits.setdefault(key,{})
            for point in curve:
                t=float((point-a[e])@direction[e]/length[e]);values[index(point)]=np.clip(t,0,1)
        patch_ids=(int(edgeids[i]),int(edgeids[j]));owner=min(patch_ids,key=lambda k:(patches[k].source_point_count,k))
        record={'kind':'bridge','bridge_type':'mesh_boundary','patch_ids':patch_ids,'owner_id':owner,
                'side_points':{patch_ids[0]:aa,patch_ids[1]:bb},'points':(aa+bb)/2,'endpoints':(aa+bb)/2}
        records.append(record)
        for polygon in ([aa[0],aa[1],bb[1]],[aa[0],bb[1],bb[0]]):
            points=np.asarray(polygon);normal=np.cross(points[1]-points[0],points[2]-points[0])
            if np.linalg.norm(normal)<1e-12:continue
            if normal@patches[owner].normal_at(patches[owner].project(points.mean(axis=0)))<0:points=points[[0,2,1]]
            indices=[index(p) for p in points]
            if len(set(indices))==3:strips.append((indices,owner))
    newfaces=[];newlabels=[];newparts=[];newflags=[]
    for face,label,part,flag in zip(tri,labels,parts,flags):
        ring=[]
        for x,y in zip(face,np.roll(face,-1)):
            ring.append(int(x));key=tuple(sorted((int(x),int(y))))
            extra=[(t,i) for i,t in splits.get(key,{}).items() if i not in (x,y) and 1e-10<t<1-1e-10]
            extra.sort(reverse=bool(x>y));ring.extend(i for _,i in extra)
        if len(ring)==3:
            newfaces.append(ring);newlabels.append(label);newparts.append(part);newflags.append(flag);continue
        coords=np.asarray(newvertices)[ring];patch=patches[int(label)];uv=patch.project(coords)
        polygon=Polygon(uv)
        if not polygon.is_valid:
            return vertices,faces,labels,parts,flags,[],0
        tree=cKDTree(uv)
        for triangle in shapely.constrained_delaunay_triangles(polygon).geoms:
            ix=tree.query(np.asarray(triangle.exterior.coords)[:3])[1];newface=[ring[int(k)] for k in ix]
            p=np.asarray(newvertices)[newface]
            if np.cross(p[1]-p[0],p[2]-p[0])@patch.normal_at(patch.project(p.mean(axis=0)))<0:newface=newface[::-1]
            newfaces.append(newface);newlabels.append(label);newparts.append(part);newflags.append(flag)
    for face,owner in strips:
        newfaces.append(face);newlabels.append(owner);newparts.append(-1 if patches[owner].part_id is None else patches[owner].part_id);newflags.append(True)
    f=np.asarray(newfaces)
    _,counts=np.unique(np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0,return_counts=True)
    if np.any(counts>2):return vertices,faces,labels,parts,flags,[],0
    return newvertices,newfaces,newlabels,newparts,newflags,records,len(strips)
