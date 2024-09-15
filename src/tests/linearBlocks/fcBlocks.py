"""
This script contains the tests for the Generic, Residual and Exponential Fully Connected Blocks 
"""

import random, torch

import torch.nn.backends
from tqdm import tqdm

import mypt.code_utilities.pytorch_utilities as pu

from mypt.linearBlocks import fully_connected_blocks as fcb, fc_block_components as fcbc


#################### Generic & exponential ####################

def _network_pass(in_features:int, 
                  out_features: int,
                  fc: fcbc.FullyConnectedBlock):

    batch_size = 10
    x = torch.randn(batch_size, in_features)
    loss_obj = torch.nn.MSELoss()
    # forward pass
    
    output = fc.forward(x)
    loss_obj.forward(output, torch.zeros(batch_size, out_features)).backward()
    # the code above should not raise an error

    # move everything to 'gpu' if avaialable
    device = pu.get_default_device()

    if 'cuda' not in device:
        return
    
    x = x.to(device)
    fc = fc.to(device)    
    loss_obj = loss_obj.to(device)

    output = fc.forward(x)
    loss_obj.forward(output, torch.zeros(batch_size, out_features).to(device)).backward()


def test_generic_fc_block(num_tests: int = 10 ** 3):

    for _ in tqdm(range(num_tests)):
        n = random.randint(1, 20)
        units = [random.randint(25, 1000) for j in range(n + 1)]
        
        # test with dropout 
        dr = [random.random() for j in range(n - 1)]

        fc = fcb.GenericFCBlock(output=units[-1], 
                                in_features=units[0],
                                num_layers=n,
                                units=units,
                                activation='relu',
                                dropout=dr)

        # make sure there are exacty n linear layers and n - 1 Relu Layers
        layers = pu.iterate(fc)
        
        fc_count = 0
        relu_count = 0
        dropout_count = 0

        for lr in layers:
            fc_count += int(isinstance(lr, torch.nn.Linear))
            relu_count += int(isinstance(lr, torch.nn.ReLU))
            dropout_count += int(isinstance(lr, torch.nn.Dropout))

        assert fc_count == n, f"Expecting {n} linear layers. Found: {fc_count}"
        assert relu_count == n - 1, f"Expecting {n - 1} Relu layers. Found: {relu_count}"
        assert dropout_count == n - 1, f"Expecting {n - 1} dropout layers. Found: {dropout_count}"


        # let's make sure the forward pass and backward passes work as expected
        _network_pass(in_features=units[0], out_features=units[-1], fc=fc)

        # without dropout
        fc = fcb.GenericFCBlock(output=units[-1], 
                                in_features=units[0],
                                num_layers=n,
                                units=units,
                                activation='tanh',
                                dropout=None)

        # make sure there are exacty n linear layers and n - 1 Relu Layers
        layers = pu.iterate(fc)
        
        fc_count = 0
        act_count = 0
        dropout_count = 0

        for lr in layers:
            fc_count += int(isinstance(lr, torch.nn.Linear))
            act_count += int(isinstance(lr, torch.nn.Tanh))
            dropout_count += int(isinstance(lr, torch.nn.Dropout))

        assert fc_count == n, f"Expecting {n} linear layers. Found: {fc_count}"
        assert act_count == n - 1, f"Expecting {n - 1} Relu layers. Found: {act_count}"
        assert dropout_count == 0, f"Expecting {0} dropout layers. Found: {dropout_count}"

        _network_pass(in_features=units[0], out_features=units[-1], fc=fc)


def test_exponential_fc_block(num_tests:int = 10 ** 3):
    for _ in tqdm(range(num_tests)):
        n = random.randint(1, 20)        
        
        d1, d2 = random.randint(10, 100), random.randint(500, 1500)
        in_f, out_f  = max(d1, d2), min(d1, d2)

        # test with dropout 
        dr = [random.random() for j in range(n - 1)]

        fc = fcb.ExponentialFCBlock(output=out_f, 
                                in_features=in_f,
                                num_layers=n,
                                activation='relu',
                                dropout=dr)

        # make sure there are exacty n linear layers and n - 1 Relu Layers
        layers = pu.iterate(fc)
        
        fc_count = 0
        relu_count = 0
        dropout_count = 0

        for lr in layers:
            fc_count += int(isinstance(lr, torch.nn.Linear))
            relu_count += int(isinstance(lr, torch.nn.ReLU))
            dropout_count += int(isinstance(lr, torch.nn.Dropout))

        assert fc_count == n, f"Expecting {n} linear layers. Found: {fc_count}"
        assert relu_count == n - 1, f"Expecting {n - 1} Relu layers. Found: {relu_count}"
        assert dropout_count == n - 1, f"Expecting {n - 1} dropout layers. Found: {dropout_count}"


        # let's make sure the forward pass and backward passes work as expected
        _network_pass(in_features=in_f, out_features=out_f, fc=fc)

        # without dropout
        fc = fcb.ExponentialFCBlock(output=out_f, 
                                in_features=in_f,
                                num_layers=n,
                                activation='tanh',
                                dropout=None)

        # make sure there are exacty n linear layers and n - 1 activation layers
        layers = pu.iterate(fc)
        
        fc_count = 0
        act_count = 0
        dropout_count = 0

        for lr in layers:
            fc_count += int(isinstance(lr, torch.nn.Linear))
            act_count += int(isinstance(lr, torch.nn.Tanh))
            dropout_count += int(isinstance(lr, torch.nn.Dropout))

        assert fc_count == n, f"Expecting {n} linear layers. Found: {fc_count}"
        assert act_count == n - 1, f"Expecting {n - 1} Relu layers. Found: {act_count}"
        assert dropout_count == 0, f"Expecting {0} dropout layers. Found: {dropout_count}"

        _network_pass(in_features=in_f, out_features=out_f, fc=fc)


#################### Residual ####################

def _residual_network_pass(
                  in_features:int, 
                  out_features: int,
                  rfc: fcbc.ResidualLinearBlock):
    
    classifier_layers, adaptive_layer = pu.iterate(rfc.classifier), pu.iterate(rfc.adaptive_layer)

    initial_w_main, initial_w_al = (
                                    [l.weight.detach() if hasattr(l, 'weight')  else None for l in classifier_layers], 
                                    [l.weight.detach() if hasattr(l, 'weight') else None for l in adaptive_layer]
                                )

    batch_size = 1000
    x = torch.randn(batch_size, in_features)
    loss_obj = torch.nn.MSELoss(reduction='sum')
    # forward pass
    
    output = rfc.forward(x)
    loss = torch.sum(loss_obj.forward(output, torch.zeros(batch_size, out_features)))
    loss.backward()

    # make sure the values change
    classifier_layers, adaptive_layer= pu.iterate(rfc.classifier), pu.iterate(rfc.adaptive_layer)
    w_main, w_al = (
                    [l.weight.detach() if hasattr(l, 'weight')  else None for l in classifier_layers], 
                    [l.weight.detach() if hasattr(l, 'weight') else None for l in adaptive_layer]
                    )

    for index, (i_l, l) in enumerate(zip(initial_w_main, w_main)):
        i_w, w = i_l, l
        
        if isinstance(classifier_layers[index], torch.nn.BatchNorm1d) or i_w is None:
            continue

        if torch.allclose(i_w, w) :
            raise ValueError(f"the backward pass did not affect the following weight")

    for i_l, l in zip(initial_w_al, w_al):
        i_w, w = i_l, l

        if isinstance(adaptive_layer[index], torch.nn.BatchNorm1d) or i_w is None:
            continue

        if torch.allclose(i_w, w):
            raise ValueError(f"the backward pass did not affect the weights of the adaptive layer")


    device = pu.get_default_device()

    if 'cuda' not in device:
        return
    

    batch_size = 10
    x = torch.randn(batch_size, in_features)

    x = x.to(device)
    rfc = rfc.to(device)    
    loss_obj = loss_obj.to(device)

    output = rfc.forward(x)
    loss =torch.sum(loss_obj.forward(output, torch.zeros(batch_size, out_features).to(device)))
    loss.backward()


def _test_residual_fc_block(num_tests:int = 10 ** 3):
    for _ in tqdm(range(num_tests)):
        n = random.randint(1, 20)        


        n = random.randint(1, 20)
        units = [random.randint(25, 1000) for j in range(n + 1)]
        
        # test with dropout 
        dr = [random.random() for j in range(n)]


        rfc = fcbc.ResidualLinearBlock(output=units[-1], 
                                in_features=units[0],
                                num_layers=n,
                                units=units,
                                activation='relu',
                                dropout=dr)


        # make sure there are exacty n linear layers and n - 1 Relu Layers
        layers = pu.iterate(rfc)
        
        fc_count = 0
        relu_count = 0
        dropout_count = 0

        for lr in layers:
            fc_count += int(isinstance(lr, torch.nn.Linear))
            relu_count += int(isinstance(lr, torch.nn.ReLU))
            dropout_count += int(isinstance(lr, torch.nn.Dropout))

        assert fc_count == n, f"Expecting {n} linear layers. Found: {fc_count}"
        assert relu_count == n, f"Expecting {n} Relu layers. Found: {relu_count}"
        assert dropout_count == n, f"Expecting {n - 1} dropout layers. Found: {dropout_count}"

        # let's make sure the forward pass and backward passes work as expected
        _residual_network_pass(in_features=units[0], out_features=units[-1], rfc=rfc)

        # without dropout
        rfc = fcbc.ResidualLinearBlock(output=units[-1], 
                                in_features=units[0],
                                num_layers=n,
                                units=units,
                                activation='tanh',
                                dropout=None)

        # make sure there are exacty n linear layers and n - 1 Relu Layers
        layers = pu.iterate(rfc)
        
        fc_count = 0
        act_count = 0
        dropout_count = 0

        for lr in layers:
            fc_count += int(isinstance(lr, torch.nn.Linear))
            act_count += int(isinstance(lr, torch.nn.Tanh))
            dropout_count += int(isinstance(lr, torch.nn.Dropout))

        assert fc_count == n, f"Expecting {n} linear layers. Found: {fc_count}"
        assert act_count == n, f"Expecting {n} Relu layers. Found: {act_count}"
        assert dropout_count == 0, f"Expecting {0} dropout layers. Found: {dropout_count}"

        _network_pass(in_features=units[0], out_features=units[-1], fc=rfc)


    pass

if __name__ == '__main__':
    # test_generic_fc_block()
    # test_exponential_fc_block()
    _test_residual_fc_block(10)
    pass