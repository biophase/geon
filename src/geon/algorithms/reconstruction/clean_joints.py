"""Clean observed face outlines before coordinating their shared joints.

The input patches are fitted surfaces. This stage changes only bounded outlines,
never fits or source labels. Near-parallel planes use complete cleaned boundary
segments rather than their poorly conditioned infinite-plane intersection.
"""
from copy import copy
from time import perf_counter

import numpy as np
import shapely
from scipy.spatial import cKDTree


def _topology(geometry):
    parts=list(shapely.get_parts(geometry))
    return len(parts), sum(len(p.interiors) for p in parts)


def clean_footprints(patches, cfg, log):
    """Simplify with explicit bidirectional deviation and topology checks."""
    before=after=0
    maximum=0.
    for number,patch in enumerate(patches.values()):
        if number%16==0:log(f'Clean outlines: {number+1}/{len(patches)}')
        original=patch.footprint
        patch._clean_observed_footprint=original
        before+=shapely.get_num_coordinates(original)
        selected=original
        for factor in (1., .5, .25, .125):
            candidate=original.simplify(cfg.simplify_tolerance*factor,preserve_topology=True)
            if not candidate.is_valid or _topology(candidate)!=_topology(original):continue
            distance=float(shapely.hausdorff_distance(original.boundary,candidate.boundary,densify=.25))
            if distance<=cfg.simplify_tolerance+1e-9:
                selected=candidate;maximum=max(maximum,distance);break
        patch.footprint=patch.polygon=selected
        patch.locked=[]
        after+=shapely.get_num_coordinates(selected)
    log(f'Clean outlines: {before:,} to {after:,} vertices; maximum boundary deviation {maximum:.6g}; holes preserved')


def _segments(patch):
    """World endpoints and directions into each polygon, including holes."""
    segments=[];inward=[]
    for polygon in shapely.get_parts(patch.footprint):
        for hole,ring in [(False,polygon.exterior)]+[(True,r) for r in polygon.interiors]:
            uv=np.asarray(ring.coords)[:-1]
            delta=np.roll(uv,-1,axis=0)-uv
            area=np.sum(uv[:,0]*np.roll(uv[:,1],-1)-uv[:,1]*np.roll(uv[:,0],-1))
            sign=(1 if area>0 else -1)*(-1 if hole else 1)
            length=np.linalg.norm(delta,axis=1)
            valid=length>1e-10
            direction=sign*np.column_stack((-delta[:,1],delta[:,0]))/np.maximum(length[:,None],1e-20)
            xyz=patch.lift(uv)
            segments.extend(np.stack((xyz,np.roll(xyz,-1,axis=0)),axis=1)[valid])
            inward.extend((direction@patch.basis.T)[valid])
    return np.asarray(segments),np.asarray(inward)


def planar_border_proposals(a,b,cfg,cache=None):
    """Match whole overlapping straight boundary segments on adjacent planes."""
    cache={} if cache is None else cache
    for p in (a,b):
        if p.id not in cache:cache[p.id]=_segments(p)
    aa,ain=cache[a.id];bb,binward=cache[b.id]
    if not len(aa) or not len(bb):return []
    bdelta=bb[:,1]-bb[:,0];blength=np.linalg.norm(bdelta,axis=1)
    centers=bb.mean(axis=1);tree=cKDTree(centers)
    tolerance=cfg.coplanar_boundary_tolerance
    result=[];seen=set()
    for segment,inward in zip(aa,ain):
        vector=segment[1]-segment[0];length=np.linalg.norm(vector);tangent=vector/length
        candidates=tree.query_ball_point(segment.mean(axis=0),length/2+blength.max()/2+tolerance)
        for index in candidates:
            other=bb[index];other_delta=bdelta[index]
            if abs(tangent@(other_delta/blength[index]))<np.cos(np.deg2rad(15)):continue
            if inward@binward[index]>=-.5:continue
            projection=(other-segment[0])@tangent
            lo=max(0.,float(projection.min()));hi=min(length,float(projection.max()))
            if hi-lo<cfg.min_edge_length:continue
            aside=segment[0]+np.array([lo,hi])[:,None]*tangent
            parameter=(aside-other[0])@other_delta/(blength[index]**2)
            bside=other[0]+np.clip(parameter,0,1)[:,None]*other_delta
            if np.linalg.norm(aside-bside,axis=1).max()>tolerance:continue
            if (np.abs((aside-b.center)@b.normal).max()>cfg.coplanar_plane_tolerance or
                    np.abs((bside-a.center)@a.normal).max()>cfg.coplanar_plane_tolerance):continue
            common=(aside+bside)/2
            key=tuple(np.round(common,8).ravel())
            if key in seen:continue
            seen.add(key)
            if cfg.coplanar_join_mode=='snap':
                result.append(dict(patch_ids=(a.id,b.id),kind='coplanar',points=common,endpoints=common.copy()))
            else:
                observed={a.id:aside.copy(),b.id:bside.copy()}
                # One common chart line is projected onto both unchanged planes.
                # Keep the two sides distinct when the planes coincide but the
                # observed outlines have a real gap, so it remains a bridge.
                separation=max(np.abs((common-a.center)@a.normal).max(),np.abs((common-b.center)@b.normal).max())
                if cfg.regularize_coplanar_borders and separation>1e-7:
                    aside=a.lift(a.project(common));bside=b.lift(b.project(common))
                gap0=aside[0]-bside[0];change=aside[1]-bside[1]-gap0
                fraction=-float(gap0@change)/max(float(change@change),1e-30)
                if 1e-8<fraction<1-1e-8 and np.linalg.norm(gap0+fraction*change)<1e-7:
                    corner=(aside[0]+fraction*(aside[1]-aside[0])+bside[0]+fraction*(bside[1]-bside[0]))/2
                    common=np.vstack((common[0],common[0]+fraction*(common[1]-common[0]),common[1]))
                    aside=np.vstack((aside[0],corner,aside[1]));bside=np.vstack((bside[0],corner,bside[1]))
                    observed={label:np.vstack((c[0],c[0]+fraction*(c[1]-c[0]),c[1])) for label,c in observed.items()}
                result.append(dict(patch_ids=(a.id,b.id),kind='bridge',bridge_type='coplanar',
                    points=common,endpoints=common[[0,-1]].copy(),side_points={a.id:aside,b.id:bside},
                    observed_side_points=observed,
                    owner_id=min((a.id,b.id),key=lambda i:(a.source_point_count if i==a.id else b.source_point_count,i))))
    return result


def coordinate_endpoints(proposals,patches,cfg):
    """Coordinate nearby incident endpoints with bounded, non-chaining groups."""
    if not proposals:return 0
    xyz=np.concatenate([e['points'][[0,-1]] for e in proposals])
    tree=cKDTree(xyz);used=set();changed=0;support={}
    limit=min(cfg.edge_tolerance,
              cfg.corner_tolerance if cfg.corner_tolerance is not None else cfg.edge_tolerance)
    for start in range(len(xyz)):
        if start in used:continue
        group=[start];labels=set(proposals[start//2]['patch_ids'])
        for other in sorted(tree.query_ball_point(xyz[start],limit)):
            if other==start or other in used or other//2 in {i//2 for i in group}:continue
            other_labels=set(proposals[other//2]['patch_ids'])
            if not labels&other_labels:continue
            if np.max(np.linalg.norm(xyz[group]-xyz[other],axis=1))>limit:continue
            group.append(other);labels|=other_labels
        if len(group)<2:continue
        if any(patches[i].kind!='plane' for i in labels):continue
        anchor=xyz[group].mean(axis=0)
        planes=[patches[i] for i in sorted(labels)]
        normals=np.stack([p.normal for p in planes])
        offsets=np.array([p.normal@(p.center-anchor) for p in planes])
        correction=np.linalg.lstsq(normals,offsets,rcond=1e-3)[0]
        target=anchor+correction
        if np.linalg.norm(target-xyz[group],axis=1).max()>limit:continue
        supported=True
        for p in planes:
            if p.id not in support:support[p.id]=cKDTree(p.points)
            if (support[p.id].query(target)[0]>cfg.adjacency_radius or
                    p.footprint.boundary.distance(shapely.Point(p.project(target)))>cfg.edge_tolerance):
                supported=False;break
        if not supported:continue
        # Analytic joints must remain on both surfaces; parallel bridge endpoints
        # may retain separate projections onto the incident planes.
        if any(proposals[i//2].get('kind')!='bridge' and
               any(abs(patches[j].normal@(target-patches[j].center))>1e-7
                   for j in proposals[i//2]['patch_ids']) for i in group):continue
        for i in group:
            edge=proposals[i//2];end=0 if i%2==0 else -1
            if np.linalg.norm(target-edge['points'][-1 if end==0 else 0])<cfg.min_edge_length:break
        else:
            for i in group:
                edge=proposals[i//2];end=0 if i%2==0 else -1
                edge['points'][end]=target
                edge['endpoints']=edge['points'][[0,-1]].copy()
                if 'side_points' in edge:
                    for label,curve in edge['side_points'].items():
                        p=patches[label];curve[end]=p.lift(p.project(target))
                used.add(i)
            changed+=1
    return changed


def _clip_planar_joint(curve,a,b,cfg,cache):
    """Keep analytic endpoints within the complete cleaned border runs.

    Radius support can extend a line past a face corner by edge_tolerance,
    creating a triangular tooth. Endpoint clipping along the exact intersection
    line prevents this without moving the joint off either fitted plane.
    """
    if len(curve)!=2:return [curve]
    origin=curve[0];vector=curve[1]-origin;length=np.linalg.norm(vector)
    if length<=0:return []
    tangent=vector/length;intervals=[(0.,length)]
    for patch in (a,b):
        if patch.id not in cache:cache[patch.id]=_segments(patch)
        segments,_=cache[patch.id];delta=segments[:,1]-segments[:,0]
        norm=np.linalg.norm(delta,axis=1)
        aligned=np.abs(delta@tangent)/np.maximum(norm,1e-20)>=np.cos(np.deg2rad(30))
        parameter=(segments-origin)@tangent
        distance=np.linalg.norm(segments-origin-parameter[:,:,None]*tangent,axis=2)
        aligned &= np.max(distance,axis=1)<=cfg.edge_tolerance
        supported=sorted(np.sort(parameter[aligned],axis=1).tolist())
        merged=[]
        for lo,hi in supported:
            if merged and lo<=merged[-1][1]+1e-8:merged[-1][1]=max(merged[-1][1],hi)
            else:merged.append([lo,hi])
        intervals=[(max(lo,x),min(hi,y)) for lo,hi in intervals for x,y in merged
                   if min(hi,y)-max(lo,x)>=cfg.min_edge_length]
        if not intervals:return []
    return [origin+np.array([lo,hi])[:,None]*tangent for lo,hi in intervals]


def _observed_curve_bound(patch,curve,cfg,cache):
    """Conservative sampled distance bound against the original observed border.

    Distance to a closed set is 1-Lipschitz. Reserving half the sampling interval
    bounds the unsampled points as well, preventing cleanup/joint drift stacking.
    """
    if patch.id not in cache:
        segments=[]
        for polygon in shapely.get_parts(patch._clean_observed_footprint):
            for ring in (polygon.exterior,*polygon.interiors):
                nodes=np.asarray(ring.coords)
                segments.extend(np.stack((nodes[:-1],nodes[1:]),axis=1))
        cache[patch.id]=shapely.STRtree(shapely.linestrings(np.asarray(segments)))
    spacing=min(cfg.support_step,cfg.edge_tolerance*.1)
    chunks=[];margin=0.
    for a,b in zip(curve[:-1],curve[1:]):
        length=np.linalg.norm(b-a);count=max(1,int(np.ceil(length/spacing)))
        margin=max(margin,length/count/2)
        chunks.append(a+np.linspace(0,1,count+1)[:,None]*(b-a))
    if not chunks:return False
    samples=np.concatenate(chunks)
    _,distances=cache[patch.id].query_nearest(shapely.points(samples),return_distance=True,all_matches=False)
    return float(distances.max())+margin<=cfg.edge_tolerance+1e-9


def reconstruct_joints(patches,candidate_pairs,cfg,diagnostics,log):
    """Return shared seams and bridges, committing only valid two-face edits."""
    from .core import (_intersection_segments,_coplanar_curves,_local_edge,_edge_points,
                       _batch_boundary,_preserves_local_locks,_bridge_is_valid,
                       _consistent_vertex_targets,_reconcile_corners)
    from .primitives import curve_in_patch
    started=perf_counter()
    clean_footprints(patches,cfg,log)
    proposals=[];cache={};border_cache={}
    for number,(i,j) in enumerate(candidate_pairs):
        pair_started=perf_counter()
        if number%16==0:log(f'Clean joints: proposing pair {number+1}/{len(candidate_pairs)}')
        a,b=patches[i],patches[j]
        near_parallel=(a.kind==b.kind=='plane' and abs(a.normal@b.normal)>=np.cos(np.deg2rad(cfg.coplanar_angle_degrees)))
        if near_parallel and cfg.join_coplanar:
            proposals.extend(planar_border_proposals(a,b,cfg,cache))
            continue
        for curve in _intersection_segments(a,b,cfg):
            curves=_clip_planar_joint(curve,a,b,cfg,cache) if a.kind==b.kind=='plane' else [curve]
            for joint in curves:
                proposals.append(dict(patch_ids=(i,j),points=joint,endpoints=joint[[0,-1]].copy()))
        if cfg.join_coplanar and (a.kind!='plane' or b.kind!='plane'):
            for edge in _coplanar_curves(a,b,cfg,border_cache=border_cache):
                if cfg.coplanar_join_mode=='bridge':
                    edge['kind']='bridge';edge['owner_id']=min((i,j),key=lambda k:(patches[k].source_point_count,k))
                proposals.append(edge)
        if perf_counter()-pair_started>=1:
            log(f'Clean joints: pair {i}/{j} completed in {perf_counter()-pair_started:.3f}s')
    analytic=[e for e in proposals if e.get('kind') not in ('bridge','coplanar')]
    _reconcile_corners(patches,analytic,cfg,log)
    coordinated=coordinate_endpoints(proposals,patches,cfg)
    log(f'Clean joints: {len(proposals)} complete-chain proposals; {coordinated} coordinated endpoint groups')
    shared=[];bridges=[];incident={i:[] for i in patches};rejected=0;observed_cache={};rematched=0
    claimed_cache={};retry_cache={}
    def unclaimed(patch,curve):
        edges=incident[patch.id]
        if not edges:return True
        cached=claimed_cache.get(patch.id)
        if cached is None:
            cached=dict(count=0,buffers=[],indexed=0,tree=None)
            claimed_cache[patch.id]=cached
        for e in edges[cached['count']:]:
            line=shapely.LineString(curve_in_patch(patch,_edge_points(e)))
            cached['buffers'].append(line.buffer(2*cfg.weld_tolerance,cap_style='flat'))
        cached['count']=len(edges)
        # Query only nearby occupied intervals. Rebuilding a union of all prior
        # seams for every proposal becomes quadratic on fragmented curved faces.
        if cached['tree'] is None or len(cached['buffers'])-cached['indexed']>=32:
            cached['tree']=shapely.STRtree(cached['buffers'])
            cached['indexed']=len(cached['buffers'])
        line=shapely.LineString(curve)
        indices=cached['tree'].query(line)
        nearby=[cached['buffers'][i] for i in indices]
        nearby.extend(g for g in cached['buffers'][cached['indexed']:] if g.intersects(line))
        return not nearby or line.intersection(shapely.union_all(nearby)).length<=20*cfg.weld_tolerance
    def validate(trial):
        edits=[]
        for label in trial['patch_ids']:
            patch=patches[label];local=_local_edge(trial,label)
            curve=curve_in_patch(patch,_edge_points(local))
            # A boundary interval already joins two faces. A second strip on
            # that interval would create overlapping/non-manifold triangles.
            if not unclaimed(patch,curve):return []
            if not _observed_curve_bound(patch,curve,cfg,observed_cache):return []
            if not _consistent_vertex_targets(patch,incident[label]+[local],cfg.weld_tolerance):return []
            current=copy(patch);current.footprint=patch.polygon
            geometry=_batch_boundary(current,[local],cfg)
            tolerance=cfg.coplanar_boundary_tolerance if local.get('kind')=='coplanar' else cfg.edge_tolerance
            if geometry is None or _topology(geometry)!=_topology(patch.polygon):return []
            if not _preserves_local_locks(patch,geometry,curve_in_patch(patch,_edge_points(local)),tolerance):return []
            edits.append((patch,geometry,local))
        if trial.get('kind')=='bridge' and not _bridge_is_valid(trial,patches,{p.id:g for p,g,_ in edits},cfg):return []
        return edits
    # Actual intersections take precedence over narrow transition strips.
    proposals.sort(key=lambda e:(e.get('kind')=='bridge',-np.linalg.norm(np.diff(e['points'],axis=0),axis=1).sum()))
    for number,edge in enumerate(proposals):
        if number%32==0:log(f'Clean joints: validating {number+1}/{len(proposals)}; accepted {len(shared)+len(bridges)}')
        attempts=[edge]
        if 'observed_side_points' in edge:
            attempts.append(dict(edge,side_points=edge['observed_side_points']))
        for trial in attempts:
            edits=validate(trial)
            if len(edits)==2:edge=trial;break
        # Earlier perpendicular joints can trim a corner of an otherwise valid
        # long border. Re-match that same complete chain on the current outlines
        # instead of discarding its full length because an endpoint moved.
        if not edits and all(patches[k].kind=='plane' for k in edge['patch_ids']):
            a,b=[copy(patches[k]) for k in edge['patch_ids']]
            a.footprint=a.polygon;b.footprint=b.polygon
            if edge.get('kind') in ('bridge','coplanar'):
                version=(len(incident[a.id]),len(incident[b.id]))
                cached=retry_cache.get((a.id,b.id))
                if cached is None or cached[0]!=version:
                    free={}
                    for p in (a,b):
                        segments,inwards=_segments(p)
                        keep=np.array([unclaimed(p,p.project(c)) for c in segments],dtype=bool)
                        free[p.id]=(segments[keep],inwards[keep])
                    cached=(version,planar_border_proposals(a,b,cfg,free))
                    retry_cache[(a.id,b.id)]=cached
                retries=cached[1]
            else:
                retries=[dict(edge,points=c,endpoints=c[[0,-1]].copy())
                         for c in _clip_planar_joint(edge['points'],a,b,cfg,{})]
            original=shapely.LineString(a.project(edge['points']))
            for retry in retries:
                # Only retry this proposal, not other accepted chains of a pair.
                if shapely.hausdorff_distance(original,shapely.LineString(a.project(retry['points'])))>2*cfg.edge_tolerance:continue
                variants=[retry]
                if 'observed_side_points' in retry:
                    variants.append(dict(retry,side_points=retry['observed_side_points']))
                for trial in variants:
                    edits=validate(trial)
                    if edits:edge=trial;rematched+=1;break
                if edits:break
        if len(edits)!=2:rejected+=1;continue
        for patch,geometry,local in edits:
            patch.polygon=geometry;incident[patch.id].append(local)
            uv=curve_in_patch(patch,_edge_points(local))
            patch.locked.extend(uv[k:k+2] for k in range(len(uv)-1))
        (bridges if edge.get('kind')=='bridge' else shared).append(edge)
    for patch in patches.values():patch.footprint=patch.polygon
    message=f'Clean joints: accepted {len(shared)} shared seams and {len(bridges)} bridges; rematched {rematched}; rejected {rejected}; {perf_counter()-started:.3f}s'
    diagnostics.append(message);log(message)
    return shared,bridges
