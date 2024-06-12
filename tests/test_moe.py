import os
os.environ['TYPECHECK'] = 'True'

import torch
import pytest

from alphafold3_pytorch import MoE, MLP, MoEGate

def test_mlp():
    x = torch.rand(2, 32, 64)
    net = MLP(64)
    out = net(x)
    assert x.shape == out.shape
    
def test_moe_gate():
    x = torch.rand(2, 32, 64)
    net = MoEGate(64)
    topk_idx, topk_weight, aux_loss = net(x)
    print(topk_idx.shape, topk_weight.shape, aux_loss.shape)
    
def test_moe():
    x = torch.rand(2, 32, 64)
    net = MoE(64)
    out = net(x)
    assert x.shape == out.shape
    
def test_moe_eval():
    x = torch.rand(2, 32, 64)
    net = MoE(64)
    net.eval()
    out = net(x)
    assert x.shape == out.shape
    
if __name__ == '__main__':
    test_moe_gate()