"""
This script is written to test losses: 
So far: 

1. Test the vectorized implementation of Simclr Loss against the naive implementation
"""

import random, torch
import numpy as np

from tqdm import tqdm

from mypt.losses.simclr_loss import SimClrLoss, _SimClrLossNaive
import mypt.code_utilities.pytorch_utilities as pu


def test_sim_clr_loss():
    pu.seed_everything(0)
    for s in ['dot', 'cos']:
        for t in tqdm(np.linspace(1, 100, 51), desc=f'testing with different temperatures with the metric: {s}'):
            loss1, loss2 = SimClrLoss(temperature=t, similarity=s), _SimClrLossNaive(temperature=t, similarity=s)

            for _ in range(1000):
                n = 2 * random.randint(10, 100)
                dim = random.randint(10, 100)    

                if s == 'dot':
                    x = torch.randn(n, dim) / 4 # using exponential with dot product usually is very numerically unstable... (using x with small norms)
                else:
                    x = torch.randint(low=0, high=10, size=(n, dim)) # using vectors with larger norms for cosine similarity

                l1 = loss1.forward(x)
                l2 = loss2.forward(x)

                assert torch.allclose(l1, l2), "Make sure the outputs of the two functions are close !!!"
        

if __name__ == '__main__':
    test_sim_clr_loss()
