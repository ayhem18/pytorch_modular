"""
This script is designed to test implementations of similarities and distances: 

1. Cosine Similarity
2. Euclidean Distance
3. MMD
"""

import random, torch, numpy as np
from tqdm import tqdm
from mypt.similarities.cosineSim import CosineSim

def _test_cos_sim(num_tests:int=10 ** 4):
    
    for i in tqdm(range(num_tests)):
        n = random.randint(10, 100)
        dim = random.randint(50, 200)
        x = 100 * torch.randn((n, dim))
        y = 100 * torch.randn((n, dim))

        sims1 = CosineSim().forward(x, y)
        assert torch.all(torch.abs(sims1) <= 1), "Cosine similarities must be between -1 and 1"

        sims2 = CosineSim().forward(x, x)
        assert np.isclose(torch.max(torch.abs(sims2)).item(), 1), "Cosine similarities must be between -1 and 1"
        assert torch.allclose(torch.diag(sims2), torch.ones(len(sims2))), "CosSim(x, x) = 1"

        sims3 = CosineSim().forward(x, -x)
        assert np.isclose(torch.max(torch.abs(sims3)).item(), 1), "Cosine similarities must be between -1 and 1"
        assert torch.allclose(torch.diag(sims3), -torch.ones(len(sims3))), "CosSim(x, -x) = 1"



if __name__ == '__main__':
    _test_cos_sim()
