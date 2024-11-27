"""
This 
"""

import torch

import mypt.code_utilities.pytorch_utilities as pu

if __name__ == "__main__":
    layer = torch.nn.Linear(in_features=100, out_features=10, bias=False )
    print(pu.get_module_num_parameters(layer))

    layer = torch.nn.Linear(in_features=100, out_features=10, bias=True)
    print(pu.get_module_num_parameters(layer))

    l1, l2 = torch.nn.Linear(in_features=1000, out_features=10, bias=False), torch.nn.Linear(in_features=500, out_features=20, bias=False)

    m = torch.nn.Sequential(l1, l2)

    print(pu.get_module_num_parameters(m))
    