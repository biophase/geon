import unittest
from types import SimpleNamespace
import numpy as np
from shapely.geometry import box
from geon.algorithms.reconstruction.primitives import Cylinder,fit_cylinder,raw_intersection_curves
from geon.algorithms.reconstruction.core import Config,reconstruct,_edge_points

class PrimitiveTests(unittest.TestCase):
    def test_cylinder_chart_preserves_single_and_batched_point_shapes(self):
        cylinder=self.patch([0,0,1],[1,0,0],[0,1,0]).cylinder
        for shape in ((3,),(1,3),(2,3),(2,4,3)):
            points=np.broadcast_to([.5,0,.3],shape)
            uv=cylinder.project(points)
            self.assertEqual(uv.shape,shape[:-1]+(2,))
            self.assertEqual(cylinder.normal_at(uv).shape,shape)
            np.testing.assert_allclose(cylinder.lift(uv),points,atol=1e-12)

    def test_plane_plane_cylinder_corner_reconciliation(self):
        from geon.algorithms.reconstruction.core import Patch,_reconcile_corners
        corner=np.array([1.,0,0])
        footprint=box(0,0,1,1)
        def plane(label,basis):
            return Patch(label,None,corner,np.cross(basis[:,0],basis[:,1]),basis,
                         corner[None,:],footprint,footprint,0.)
        a=plane(0,np.array([[1.,0],[0,1],[0,0]]))
        b=plane(1,np.array([[1.,0],[0,0],[0,1]]))
        cylinder=Cylinder(np.zeros(3),np.array([0.,0,1]),np.array([1.,0,0]),
                          np.array([0.,1,0]),1.,0.,1.,False,0.)
        c=plane(2,np.eye(3)[:,:2]);c.cylinder=cylinder
        curves=[((0,1),np.array([[1.03,0,0],[1.5,0,0]])),
                ((0,2),np.array([[np.cos(.03),np.sin(.03),0],[np.cos(.5),np.sin(.5),0]])),
                ((1,2),np.array([[1.,0,.03],[1.,0,.5]]))]
        proposals=[dict(patch_ids=ids,points=curve.copy(),endpoints=curve.copy()) for ids,curve in curves]
        count=_reconcile_corners({0:a,1:b,2:c},proposals,Config(edge_tolerance=.1),lambda _:None)
        self.assertEqual(count,1)
        for edge in proposals:
            np.testing.assert_allclose(edge['points'][0],corner,atol=1e-7)

    def cylinder_points(self):
        theta,z=np.meshgrid(np.linspace(0,2*np.pi,65)[:-1],np.linspace(0,2,25))
        return np.column_stack([.5*np.cos(theta.ravel()),.5*np.sin(theta.ravel()),z.ravel()])

    def test_fitting_and_periodic_mesh(self):
        xyz=self.cylinder_points();r=reconstruct(xyz,np.ones(len(xyz),dtype=int),config=Config(alpha_radius=.12))
        self.assertEqual(r.patches[1].kind,'cylinder')
        self.assertAlmostEqual(r.patches[1].cylinder.radius,.5,places=4)
        self.assertLess(np.abs(np.linalg.norm(r.vertices[:,:2],axis=1)-.5).max(),1e-6)
        edges=np.sort(np.concatenate([r.faces[:,[0,1]],r.faces[:,[1,2]],r.faces[:,[2,0]]]),axis=1)
        unique,count=np.unique(edges,axis=0,return_counts=True)
        mid=r.vertices[unique[count==1]].mean(axis=1)
        self.assertTrue(np.all((np.abs(mid[:,2])<1e-6)|(np.abs(mid[:,2]-2)<1e-6)),mid)
        tri=r.vertices[r.faces];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1).sum()/2
        self.assertLess(abs(area-2*np.pi),.15)
        mid=tri.mean(axis=1)
        self.assertLess(np.abs(np.linalg.norm(mid[:,:2],axis=1)-.5).max(),.010001)
        # A regular grid must also limit axial edges, not only angular error.
        self.assertLessEqual(np.ptp(tri[:,:,2],axis=1).max(),.25+1e-8)

    def test_plane_cylinder_closed_join(self):
        cylinder=self.cylinder_points()
        x,y=np.meshgrid(np.linspace(-.8,.8,45),np.linspace(-.8,.8,45));x=x.ravel();y=y.ravel();keep=x*x+y*y>.52**2
        plane=np.column_stack([x[keep],y[keep],x[keep]*0+2])
        r=reconstruct(np.vstack([cylinder,plane]),np.r_[np.ones(len(cylinder),int),np.full(len(plane),2,int)],config=Config(alpha_radius=.12,edge_tolerance=.12,adjacency_radius=.12))
        self.assertTrue(r.shared_edges,r.diagnostics)
        for edge in r.shared_edges:
            curve=_edge_points(edge)
            ids=[np.argmin(np.linalg.norm(r.vertices-point,axis=1)) for point in curve]
            for label in (1,2):
                triangles=r.faces[r.face_patch_ids==label]
                for a,b in zip(ids,ids[1:]):
                    self.assertTrue(np.any(np.isin(triangles,[a,b]).sum(axis=1)==2),(label,a,b))

    def patch(self,axis,u,v):
        c=Cylinder(np.zeros(3),np.array(axis,float),np.array(u,float),np.array(v,float),.5,0,2*np.pi,True,0)
        return SimpleNamespace(cylinder=c,footprint=box(0,-1,np.pi,1))

    def test_supported_tangent_plane_cylinder_border_across_chart_wrap(self):
        from geon.algorithms.reconstruction.core import Patch,_join_coplanar
        cylinder=Cylinder(np.zeros(3),np.array([0.,0,1]),np.array([1.,0,0]),
                          np.array([0.,1,0]),1.,0.,np.pi,False,0.)
        u,v=np.meshgrid(np.linspace(0,np.pi,40),np.linspace(0,1,25))
        uv=np.column_stack((u.ravel(),v.ravel()))
        footprint=box(0,0,np.pi,1)
        a=Patch(0,None,np.zeros(3),np.array([1.,0,0]),np.eye(3)[:,:2],
                cylinder.lift(uv),footprint,footprint,0.,cylinder=cylinder)
        y,z=np.meshgrid(np.linspace(-1,-.02,30),np.linspace(0,1,25))
        points=np.column_stack((np.ones(y.size),y.ravel(),z.ravel()))
        footprint=box(-1,0,-.02,1)
        b=Patch(1,None,np.array([1.,0,0]),np.array([1.,0,0]),np.array([[0.,0],[1,0],[0,1]]),
                points,footprint,footprint,0.)
        cfg=Config(coplanar_boundary_tolerance=.06,coplanar_plane_tolerance=.03,support_step=.01)
        shared,bridges=_join_coplanar({0:a,1:b},[],[(0,1)],cfg,[],lambda _:None)
        self.assertTrue(bridges)
        self.assertGreater(sum(np.linalg.norm(np.diff(e['points'],axis=0),axis=1).sum() for e in bridges),.9)
        # Remote borders must not be joined merely because normals agree.
        b.center=b.center+np.array([.2,0,0]);b.points=b.points+np.array([.2,0,0])
        a.locked=[];b.locked=[]
        _,remote=_join_coplanar({0:a,1:b},[],[(0,1)],cfg,[],lambda _:None)
        self.assertFalse(remote)

    def test_cylinder_cylinder_analytic_curves(self):
        a=self.patch([0,0,1],[1,0,0],[0,1,0]);b=self.patch([1,0,0],[0,1,0],[0,0,1])
        curves=raw_intersection_curves(a,b,.02)
        self.assertEqual(len(curves),2)
        for curve in curves:
            xyz=curve[np.isfinite(curve).all(axis=1)]
            self.assertLess(np.abs(a.cylinder.residual(xyz)).max(),1e-7)
            self.assertLess(np.abs(b.cylinder.residual(xyz)).max(),1e-7)

    def test_plane_cylinder_generator_lines(self):
        a=self.patch([0,0,1],[1,0,0],[0,1,0]);b=SimpleNamespace(cylinder=None,normal=np.array([1,0,0]),center=np.zeros(3))
        curves=raw_intersection_curves(a,b,.02)
        self.assertEqual(len(curves),2)
        for curve in curves:self.assertLess(np.abs(curve[:,0]).max(),1e-7)

    def test_two_cylinders_join_along_a_supported_curved_edge(self):
        theta,t=np.meshgrid(np.linspace(.2,1.2,42),np.linspace(0,1,26));theta=theta.ravel();t=t.ravel()
        root=.5*np.cos(theta)
        first=np.column_stack([.5*np.cos(theta),.5*np.sin(theta),root+.02+t*(1-root-.02)])
        second=np.column_stack([root+.02+t*(1-root-.02),.5*np.sin(theta),.5*np.cos(theta)])
        xyz=np.vstack([first,second]);labels=np.repeat([1,2],len(first))
        r=reconstruct(xyz,labels,config=Config(alpha_radius=.08,cylinder_plane_trigger=.005,edge_tolerance=.1,adjacency_radius=.1))
        self.assertEqual([p.kind for p in r.patches.values()],['cylinder','cylinder'])
        self.assertTrue(r.shared_edges,r.diagnostics)
        for edge in r.shared_edges:
            curve=_edge_points(edge)
            for p in r.patches.values():self.assertLess(np.abs(p.cylinder.residual(curve)).max(),1e-6)
            indices=[np.argmin(np.linalg.norm(r.vertices-point,axis=1)) for point in curve]
            for label in edge['patch_ids']:
                faces=r.faces[r.face_patch_ids==label]
                for a,b in zip(indices,indices[1:]):
                    self.assertTrue(np.any(np.isin(faces,[a,b]).sum(axis=1)==2))

    def test_extent_mode_fills_scan_footprint_holes_on_complete_cylinder(self):
        xyz=self.cylinder_points()
        theta=np.arctan2(xyz[:,1],xyz[:,0])
        keep=~((np.abs(theta)<.3)&(xyz[:,2]>.7)&(xyz[:,2]<1.2))
        xyz=xyz[keep]
        r=reconstruct(xyz,np.ones(len(xyz),dtype=int),config=Config(alpha_radius=.08,cylinder_trim_mode='extent'))
        self.assertEqual(r.patches[1].kind,'cylinder')
        self.assertEqual(len(r.patches[1].polygon.interiors),0)
        self.assertAlmostEqual(r.patches[1].polygon.area,2*np.pi*.5*2,places=4)

if __name__=='__main__':unittest.main()
