import pytest

import numpy as np
import torch

from blocks import BiorthogonalPairs, Normalizer


def test_normalizer():
    """
    This is to verify that the normalizer works properly, i.e., 
    1) max(normalizer.forward(u)) = 1, min(normalizer.forward(u)) = 0
    2) normalizer.backward reverses normalizer.forward.
    """

    # Config
    np.random.seed(1)
    torch.manual_seed(1)

    # Generate dataset
    ns = 125
    nh = 1001
    U = 14 * torch.rand(ns, nh) - 3 
    U2 = 14 * torch.rand(2 * ns - 3, nh) - 3 

    # Setup normalizer 
    normalizer = Normalizer(U)

    # Check if normalization is effective
    Unorm = normalizer.forward(U)
    assert torch.allclose(Unorm.max() , torch.ones_like(Unorm))
    assert torch.allclose(Unorm.min(), torch.zeros_like(Unorm))

    # Check if denormalization reverses normalization
    assert torch.allclose(U2, normalizer.backward(normalizer.forward(U2)))



def test_biorthogonal_pairs():
    """
    This is to verify that:
    1) the proposed parameterization for constructing a pair of biorthogonal 
       matrices actually satisfy the constraint, up to some tolerance.
    2) the rank-greater-than-dofs exception is suitably raised.
    """

    # Config
    np.random.seed(1)
    torch.manual_seed(1)

    # Explore both the cases when min(r, n-r) = r and min(r, n-r) = n-r
    for (n, r) in zip((30, 30), (20, 10)):
        biorth_pairs = BiorthogonalPairs(n = n, r = r)
        biorth_pairs.finalize_init()
        I_appx = biorth_pairs.enc_mat @ biorth_pairs.dec_mat
        assert torch.allclose(I_appx, torch.eye(I_appx.shape[0]), atol = 1e-6)

    # When r > n
    with pytest.raises(ValueError):
        biorth_pairs = BiorthogonalPairs(n = 20, r = 25)