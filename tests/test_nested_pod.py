import copy

import numpy as np
import torch

from utils import generate_data_for_tests_suite
from NestedPOD import NestedPOD
from modules import AE, SAE, SBAE, SOAE
from activations import HypAct
from blocks import Normalizer


def test_nested_pod_dimension_consistency():
    """ 
    This is to verify consistency between the dimensions of NestedPOD's
    low-rank matrices and the weights of AE, SAE, SBAE, SOAE we assign them
    to, within the context of EYS initialization.
    """

    # Config
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    np.random.seed(1)
    torch.manual_seed(1)

    # Generate data
    utrain, _, _ = generate_data_for_tests_suite(device = device)

    # Instantiate normalizer
    normalizer = Normalizer(utrain = utrain)

    # Instantiate some activation and nested POD
    theta = 0.3
    bilipactivation = HypAct(theta) 
    bilipactivation.setup()
    red_dims = (utrain.shape[1], 40, 30, 20, 3)
    nested_pod = NestedPOD(
        normalizer.forward(utrain), 
        red_dims, 
        bilipactivation
    )
    lenWs = len(nested_pod.Ws)

    # Test dimensions for AE
    ae = AE(red_dims, bilipactivation)
    assert (lenWs == len(ae.weights_enc)) and \
        (lenWs == len(ae.weights_dec))
    for k in range(lenWs):
        W, b = nested_pod.get_affine_transform(
            level = k, 
            proj_dim = red_dims[k+1]
        )
        assert ae.weights_enc[k].weight.data.shape == W.shape 
        assert ae.weights_dec[-k-1].weight.data.shape == W.T.shape
        assert ae.weights_enc[k].bias.data.shape == ((-b[0]) @ W.T).shape
        assert ae.weights_dec[-k-1].bias.data.shape == b[0].shape


    # Test dimensions for SAE
    sae = SAE(red_dims, bilipactivation)
    assert (lenWs == len(sae.weights_enc)) and \
        (lenWs == len(sae.weights_dec))
    for k in range(lenWs):
        W, b = nested_pod.get_affine_transform(
            level = k, 
            proj_dim = red_dims[k+1]
        )
        assert sae.weights_enc[k].weight.data.shape == W.shape 
        assert sae.weights_dec[-k-1].weight.data.shape == W.T.shape
        assert sae.weights_enc[k].bias.data.shape == ((-b[0]) @ W.T).shape
        assert sae.weights_dec[-k-1].bias.data.shape == b[0].shape


    # Test dimensions for SBAE
    sbae = SBAE(red_dims, bilipactivation)
    assert (lenWs == len(sbae.weights)) and \
        (lenWs == len(sbae.weights))
    for k in range(lenWs):
        W, b = nested_pod.get_affine_transform(
            level = k, 
            proj_dim = red_dims[k+1]
        )
        assert sbae.weights[k].enc_mat.shape == W.shape 
        assert sbae.weights[k].dec_mat.shape == W.T.shape 
        assert sbae.biases[k].data.shape == b[0].shape
        

    # Test dimensions for SOAE
    soae = SOAE(red_dims, bilipactivation)
    assert (lenWs == len(soae.weights)) and \
        (lenWs == len(soae.weights))
    for k in range(lenWs):
        W, b = nested_pod.get_affine_transform(
            level = k, 
            proj_dim = red_dims[k+1]
        )
        assert soae.weights[k].weight.data.shape == W.shape 
        assert soae.biases[k].data.shape == b[0].shape


 
def test_nested_pod_encode_output_consistency():
    """ 
    This test aims to verify that, when initialized with the EYS routine,
    SAE's encoder output is suitably close to the NestedPOD's encoder output.
    """

    # Config
    np.random.seed(1)
    torch.manual_seed(1)

    # Generate data
    utrain, _, _ = generate_data_for_tests_suite(device = 'cpu')

    # Instantiate normalizer
    normalizer = Normalizer(utrain = utrain)

    # Instantiate some activation and Nested POD
    theta = 0.3
    bilipactivation = HypAct(theta) 
    bilipactivation.setup()
    red_dims = (utrain.shape[1], 10, 3, 2)
    nested_pod = NestedPOD(
        normalizer.forward(utrain), 
        red_dims, 
        bilipactivation
    )

    # Only SAE (if the test_init suite is passed there is no need to check SBAE
    # and SOAE)
    sae = SAE(red_dims, bilipactivation)
    sae.eys(nested_pod = nested_pod)
    output_sae_encoder = sae.encode(normalizer.forward(utrain))
    output_sae_encoder = output_sae_encoder.detach().cpu().numpy()

    # Nested POD
    output_nested_pod_encoder = copy.copy(
        normalizer.forward(utrain).numpy()
    )
    for k in range(nested_pod.num_levels):
        Wk = nested_pod.Ws[k][:red_dims[k+1]]
        bk = nested_pod.bs[k][0]
        output_nested_pod_encoder = (output_nested_pod_encoder - bk) @ Wk.T
        if k < nested_pod.num_levels - 1:
            output_nested_pod_encoder = nested_pod.bilipactivation.act(
                output_nested_pod_encoder
            )

    # Check 
    squared_err = np.sum(
        (output_sae_encoder - output_nested_pod_encoder)**2, 
        axis = 1
    )
    assert np.allclose(np.mean(squared_err), 0)

