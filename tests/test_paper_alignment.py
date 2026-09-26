import sys
import unittest
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(2)
from uie.models import FastUNetEnhancer
from uie.losses import ComboLoss

class PaperAlignmentTests(unittest.TestCase):
    def test_shape_range_and_existing_checkpoint(self):
        model=FastUNetEnhancer().eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()),309862)
        ck=torch.load(ROOT/"runs/uie_fastunet_base32/best.pt",map_location="cpu",weights_only=True)
        model.load_state_dict(ck.get("ema") or ck["model"],strict=True)
        with torch.no_grad():
            y=model(torch.rand(1,3,33,49))
        self.assertEqual(y.shape,(1,3,33,49))
        self.assertTrue(torch.isfinite(y).all())
        self.assertTrue(((y>=0)&(y<=1)).all())

    def test_pure_objectives(self):
        x=torch.rand(1,3,32,32,requires_grad=True); y=torch.rand_like(x)
        for objective,expected in [("l1",(x-y).abs().mean()),("mse",(x-y).square().mean())]:
            fn=ComboLoss(objective=objective)
            actual,_=fn(x,y)
            torch.testing.assert_close(actual,expected)
            self.assertIsNone(fn.perc)
        actual.backward()
        self.assertGreater(x.grad.abs().sum().item(),0)

    def test_ssim_contributes_to_gradient(self):
        x=torch.rand(1,3,32,32,requires_grad=True); y=torch.rand_like(x)
        loss,_=ComboLoss(w_l1=0,w_ssim=1,w_perc=0)(x,y)
        loss.backward()
        self.assertGreater(x.grad.abs().sum().item(),0)

if __name__ == "__main__":
    unittest.main()
