import unittest
import torch
from paper.benchmarks2026.common import ManifoldOps,PointGenerator
from paper.benchmarks2026.mgm import create_method


class SharedGeometry(unittest.TestCase):
    def test_paths(self):
        torch.set_num_threads(1)
        for kind,dim in [("sphere",3),("torus",2),("torus",7)]:
            ops=ManifoldOps(kind,dim);r=torch.Generator().manual_seed(13)
            x,y=ops.base(64,r,"cpu",torch.float64),ops.base(64,r,"cpu",torch.float64)
            t=torch.rand(64,1,generator=r,dtype=torch.float64)*.9+.05
            point,velocity=ops.path(x,y,t)
            reverse,back=ops.path(y,x,1-t)
            self.assertLess(float(ops.distance2(point,reverse).max()),1e-20)
            self.assertLess(float((velocity+back).abs().max()),1e-10)
            _,jvp=torch.func.jvp(lambda s:ops.path(x,y,s)[0],(t,),(torch.ones_like(t),))
            self.assertLess(float((velocity-jvp).abs().max()),1e-10)

    def test_antipodal_reversal(self):
        ops=ManifoldOps("sphere",3)
        x=ops.proj(torch.tensor([[1.,0,0],[1.,2.,3.]],dtype=torch.float64));y=-x
        t=torch.tensor([[.2],[.7]],dtype=torch.float64)
        a,u=ops.path(x,y,t);b,v=ops.path(y,x,1-t)
        self.assertLess(float((a-b).abs().max()),1e-12)
        self.assertLess(float((u+v).abs().max()),1e-12)
        self.assertLess(float((ops.exp(x,ops.log(x,y))-y).abs().max()),1e-12)

    def test_generator_and_mixed_derivatives(self):
        for kind,dim in [("sphere",3),("torus",7)]:
            ops=ManifoldOps(kind,dim);r=torch.Generator().manual_seed(4)
            model=PointGenerator(ops).double();z=torch.randn(32,16,dtype=torch.float64,generator=r)
            self.assertTrue(torch.allclose(model(z),ops.base_from_latent(z),atol=1e-12))
            _,d=torch.func.jvp(model,(z,),(torch.ones_like(z),))
            d.square().sum().backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_mgm_finite(self):
        for kind,dim in [("sphere",3),("torus",2)]:
            ops=ManifoldOps(kind,dim);r=torch.Generator().manual_seed(4)
            model=create_method(ops,{"critic_warmup":1,"critic_steps":2,"noise":.7},"cpu")
            for _ in range(4):
                logs=model.update(ops.base(64,r,"cpu"),r)
                self.assertTrue(torch.isfinite(logs["loss"]))
            self.assertTrue(all(torch.isfinite(p).all() for p in model.model.parameters()))


if __name__=="__main__":unittest.main()
