import numpy as np
import torch

import metrics


def test_metrics():
    """ This is to test that MSE and MRE are calculated properly."""

    Xt = torch.tensor([[1, 1, 1], [2, -1, 0]], dtype = torch.float32)
    Zt = torch.tensor([[-1, 1, 0], [1, 0, 1]], dtype = torch.float32)
    assert np.allclose(metrics.mse(utrue = Xt, upred = Zt).item(), 4)
    assert np.allclose(
        metrics.mre(utrue = Xt, upred = Zt).item(), 
        0.5 * (np.sqrt(5/3) + np.sqrt(3/5))
    )