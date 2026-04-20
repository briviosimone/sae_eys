import numpy as np
import torch

from activations import LeakyReLU, HypAct


def test_invertibility():
    """ 
    This is to verify that the bilipschitz activations employed within the 
    library are actually invertible.
    """

    xlin = torch.linspace(-100,100,10000)
    
    leakyrelu = LeakyReLU(alpha = 0.3, beta = 2)
    hypact = HypAct(alpha = 0.1)

    assert np.allclose(xlin, leakyrelu.act(leakyrelu.invact(xlin)))
    assert np.allclose(xlin, leakyrelu.invact(leakyrelu.act(xlin)))
    assert np.allclose(xlin, hypact.act(hypact.invact(xlin)))
    assert np.allclose(xlin, hypact.invact(hypact.act(xlin)))



def test_lipschitz(): 
    """ 
    To verify that the prescribed lipschitz constants for the activation \rho
    used within the library are correct, we ensure that they are suitably 
    close to max_x |\rho'(x)|.
    """

    leakyrelu = LeakyReLU(alpha = 0.3, beta = 2)
    hypact = HypAct(alpha = 0.1)

    def assert_lip(act_fun : callable, lip_exact : float):
        xlin = np.linspace(-1e8, 1e8, int(1e6))
        ylin = act_fun(xlin)
        ylin_der = np.gradient(ylin, xlin[1] - xlin[0])
        lip_act_est = np.max(np.abs(ylin_der))
        assert np.allclose(lip_act_est, lip_exact)

    assert_lip(leakyrelu.act, leakyrelu.lip_act)
    assert_lip(leakyrelu.invact, leakyrelu.lip_invact)
    assert_lip(hypact.act, hypact.lip_act)
    assert_lip(hypact.invact, hypact.lip_invact)