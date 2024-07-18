"""
# let's test the vectorized implementation against the naive one and make sure the outputs of both functions are close
"""

import random, torch
import numpy as np

from mypt.losses.simClrLoss import SimClrLoss, _SimClrLossNaive
import mypt.code_utilities.pytorch_utilities as pu


def test_sim_clr_loss():
    pu.seed_everything(0)
    for t in np.linspace(1, 10, 51):
        loss1, loss2 = SimClrLoss(temperature=t), _SimClrLossNaive(temperature=t)

        for _ in range(100):
            n = 2 * random.randint(10, 100)
            dim = random.randint(10, 100)    
            x = torch.randn(n, dim)
            l1 = loss1.forward(x)
            l2 = loss2.forward(x)

            assert torch.allclose(l1, l2), "Make sure the outputs of the two functions are close !!!"
        

if __name__ == '__main__':
    test_sim_clr_loss()
