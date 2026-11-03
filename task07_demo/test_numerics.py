"""Numerical guard regression: reject invalid values before optimizer updates."""
import unittest
import torch
from run_unet import loss,require_finite,clip_gradients
class NumericalTests(unittest.TestCase):
    def test_large_fp32_logits_have_finite_loss_and_gradient(self):
        logits=torch.tensor([4000.,-4000.,1000.]).reshape(1,3,1,1).expand(2,3,8,8).clone().requires_grad_()
        y=torch.ones((2,8,8),dtype=torch.long)
        v=loss(logits,y);require_finite(v,'loss');v.backward();require_finite(logits.grad,'gradients')
    def test_nonfinite_logits_rejected(self):
        with self.assertRaises(RuntimeError):require_finite(torch.tensor([float('inf')]),'logits')
    def test_nonfinite_gradient_does_not_update_parameters(self):
        net=torch.nn.Linear(1,1);before=net.weight.detach().clone();net.weight.grad=torch.full_like(net.weight,float('inf'))
        with self.assertRaises(RuntimeError):clip_gradients(net)
        self.assertTrue(torch.equal(before,net.weight))
if __name__=='__main__':unittest.main()
