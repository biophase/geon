import unittest
import numpy as np
from geon.algorithms.reconstruction.core import Config,reconstruct

class GapRecoveryTests(unittest.TestCase):
    def data(self):
        x,t=np.meshgrid(np.linspace(0,1,26),np.linspace(.03,1,26));x=x.ravel();t=t.ravel();angle=np.deg2rad(80)
        a=np.column_stack([x,t,t*0]);b=np.column_stack([x,t*np.cos(angle),t*np.sin(angle)])
        return np.vstack([a,b]),np.repeat([1,2],len(x))

    def test_rejected_sharp_join_can_be_bridged_without_moving_planes(self):
        xyz,ids=self.data()
        cfg=Config(alpha_radius=.07,min_plane_angle_degrees=89,bridge_remaining_gaps=True)
        r=reconstruct(xyz,ids,config=cfg)
        strips=[b for b in r.bridges if b.get('bridge_type')=='corner_gap']
        self.assertTrue(strips,r.diagnostics)
        self.assertTrue(r.face_is_bridge.any())
        for label,p in r.patches.items():
            vertices=r.vertices[np.unique(r.faces[(r.face_patch_ids==label)&~r.face_is_bridge])]
            self.assertLess(np.abs((vertices-p.center)@p.normal).max(),1e-8)
        edges=np.sort(np.concatenate([r.faces[:,[0,1]],r.faces[:,[1,2]],r.faces[:,[2,0]]]),axis=1)
        _,count=np.unique(edges,axis=0,return_counts=True)
        self.assertTrue(np.all(count<=2))

    def test_disabled_recovery_leaves_the_open_shells(self):
        xyz,ids=self.data()
        r=reconstruct(xyz,ids,config=Config(alpha_radius=.07,min_plane_angle_degrees=89,bridge_remaining_gaps=False))
        self.assertFalse(r.bridges)

    def test_small_gap_limit_does_not_invent_a_bridge(self):
        xyz,ids=self.data()
        r=reconstruct(xyz,ids,config=Config(alpha_radius=.07,min_plane_angle_degrees=89,gap_bridge_tolerance=.01))
        self.assertFalse(r.bridges)

if __name__=='__main__':unittest.main()
