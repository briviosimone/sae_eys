import numpy as np
import torch
import torch.nn as nn
from dlroms import * 
from dolfin import *

from blocks import Normalizer
from modules import AE, SAE, SBAE, SOAE
from activations import LeakyReLU, HypAct
from NestedPOD import NestedPOD
from utils import generate_data_for_tests_suite



def test_eys_init_consistency():
    """ 
    This test serves to verify that, when initialized with EYS, up to some 
    tolerance,
    1) SAE, SBAE, SOAE have the save weights and biases, according to the paper
       notation;
    2) The forward passes of SAE, SBAE, SOAE yield the same MSE.
    """

    # Config
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    np.random.seed(1)
    torch.manual_seed(1)

    # Generate data
    utrain, uval, utest = generate_data_for_tests_suite(device = device)

    # Instantiate some activation
    theta = 0.3
    bilipactivation = HypAct(theta) 
    bilipactivation.setup()
    red_dims = (utrain.shape[1], 20, 15, 3)

    # Use normalizer
    normalizer = Normalizer(utrain = utrain)

    # Instantiate NestesPOD for EYS initialization
    nested_pod = NestedPOD(
        normalizer.forward(utrain), 
        red_dims, 
        bilipactivation
    )
    
    # Initialize SAE with EYS and compute test MSE
    sae = SAE(red_dims, bilipactivation)
    sae.eys(nested_pod = nested_pod)
    sae.to(device)
    upred = normalizer.backward(sae(normalizer.forward(utest)))
    mse_test_sae = msei(euclidean)(utrue = utest, upred = upred)

    # Initialize SAE with EYS and compute test MSE
    soae = SOAE(red_dims, bilipactivation)
    soae.eys(nested_pod = nested_pod)
    soae.to(device)
    upred = normalizer.backward(soae(normalizer.forward(utest)))
    mse_test_soae = msei(euclidean)(utrue = utest, upred = upred)

    # Initialize SAE with EYS and compute test MSE
    sbae = SBAE(red_dims, bilipactivation)
    sbae.eys(nested_pod = nested_pod)
    sbae.to(device)
    upred = normalizer.backward(sbae(normalizer.forward(utest)))
    mse_test_sbae = msei(euclidean)(utrue = utest, upred = upred)

    # Check weight by weight
    for k in range(sae.n_levels):
        diff_enc_sbae = sae.weights_enc[k].weight - sbae.weights[k].enc_mat
        diff_dec_sbae = sae.weights_dec[-k-1].weight - sbae.weights[k].dec_mat
        diff_enc_soae = sae.weights_enc[k].weight - soae.weights[k].weight
        diff_dec_soae = sae.weights_dec[-k-1].weight - soae.weights[k].weight.T
        for d in (diff_enc_sbae, diff_dec_sbae, diff_enc_soae, diff_dec_soae):
            assert np.allclose((d).pow(2).sum().item(), 0)

    # Check bias by bias
    for k in range(sae.n_levels):
        sbae_enc_bias = - sbae.biases[k] @ sbae.weights[k].enc_mat.T
        soae_enc_bias = - soae.biases[k] @ soae.weights[k].weight.T
        diff_enc_sbae = sae.weights_enc[k].bias - sbae_enc_bias
        diff_dec_sbae = sae.weights_dec[-k-1].bias - sbae.biases[k]
        diff_enc_soae = sae.weights_enc[k].bias - soae_enc_bias
        diff_dec_soae = sae.weights_dec[-k-1].bias - soae.biases[k]
        for d in (diff_enc_sbae, diff_dec_sbae, diff_enc_soae, diff_dec_soae):
            assert np.allclose((d).pow(2).sum().item(), 0)
    
    # Further check using test MSE 
    assert np.allclose(mse_test_sae, mse_test_soae) and \
        np.allclose(mse_test_sae, mse_test_sbae)



def test_sbae_soae_standard_init_consistency():
    """ 
    This test serves to verify that, when initialized with the standard random 
    initialization, up to some tolerance,
    1) SBAE and SOAE have the save weights and biases, according to the paper
       notation;
    2) The forward passes of SBAE and SOAE yield the same MSE.
    """

    # Config
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    np.random.seed(1)
    torch.manual_seed(1)

    # Generate data
    utrain, uval, utest = generate_data_for_tests_suite(device = device)

    # Instantiate some activation
    theta = 0.3
    bilipactivation = HypAct(theta) 
    bilipactivation.setup()
    red_dims = (utrain.shape[1], 20, 15, 3)

    # Use normalizer
    normalizer = Normalizer(utrain = utrain)

    # Test MSE for SOAE
    np.random.seed(1)
    torch.manual_seed(1)
    soae = SOAE(red_dims, bilipactivation)
    soae.standard()
    soae.to(device)
    upred = normalizer.backward(soae(normalizer.forward(utest)))
    mse_test_soae = msei(euclidean)(utrue = utest, upred = upred)

    # Test MSE for SBAE
    np.random.seed(1)
    torch.manual_seed(1)
    sbae = SBAE(red_dims, bilipactivation)
    sbae.standard()
    sbae.to(device)
    upred = normalizer.backward(sbae(normalizer.forward(utest)))
    mse_test_sbae = msei(euclidean)(utrue = utest, upred = upred)

    # Test weight by weight
    for k in range(sbae.n_levels):
        diff_enc = sbae.weights[k].enc_mat - soae.weights[k].weight
        diff_dec = sbae.weights[k].dec_mat - soae.weights[k].weight.T
        for d in (diff_enc, diff_dec):
            assert np.allclose((d).pow(2).sum().item(), 0)

    # Test bias by bias
    for k in range(sbae.n_levels):
        diff = sbae.biases[k] - soae.biases[k]
        assert np.allclose((diff).pow(2).sum().item(), 0)

    # Further check using test MSE 
    assert np.allclose(mse_test_soae, mse_test_sbae)



def test_ae_sae_standard_init_vs_plain_torch():
    """ 
    This test serves to verify that the standard random initialization
    implemented in this library, up to some tolerance, coincides with the He 
    initialization implemented on the torch library, in the case of
    LeakyReLU_{alpha,1}.
    """

    # Config
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    np.random.seed(1)
    torch.manual_seed(1)

    # Generate data
    utrain, _, _ = generate_data_for_tests_suite(device = device)
    red_dims = (utrain.shape[1], 20, 15, 3)

    # Using LeakyReLU(\alpha,1)
    alpha = 5.0
    bilipactivation = LeakyReLU(alpha, 1) 
    bilipactivation.setup()

    # Generate skeleton using plain torch
    enc_list = [
        nn.Linear(red_dims[k], red_dims[k+1]).to(device)
        for k in range(len(red_dims) - 1)
    ]
    dec_list =  [
        nn.Linear(red_dims[-k], red_dims[-k-1]).to(device)
        for k in range(1, len(red_dims))
    ]
    
    # Test function for AE
    def test_ae():

        # initialize using plain torch and kaiming init
        np.random.seed(1)
        torch.manual_seed(1)
        n_levels = len(red_dims) - 1
        for k in range(n_levels):
            nn.init.kaiming_normal_(enc_list[k].weight, a = 1/alpha)
            nn.init.kaiming_normal_(dec_list[-k-1].weight, a = 1/alpha)

        # define and initialize AE with library
        ae = AE(red_dims, bilipactivation)
        ae.to(device)
        np.random.seed(1)
        torch.manual_seed(1)
        ae.standard()
        
        # Check
        for k in range(n_levels):
            assert torch.allclose(ae.weights_enc[k].weight, enc_list[k].weight)
            assert torch.allclose(ae.weights_dec[k].weight, dec_list[k].weight)

    # Test function for SAE
    def test_sae():
        
        # initialize using plain torch and kaiming init
        np.random.seed(1)
        torch.manual_seed(1)
        n_levels = len(red_dims) - 1
        for k in range(n_levels):
            nn.init.kaiming_normal_(enc_list[k].weight, a = alpha)
            nn.init.kaiming_normal_(dec_list[-k-1].weight, a = 1/alpha)

        # define and initialize SAE with library
        sae = SAE(red_dims, bilipactivation)
        sae.to(device)
        np.random.seed(1)
        torch.manual_seed(1)
        sae.standard()

        # Check
        for k in range(n_levels):
            assert torch.allclose(sae.weights_enc[k].weight, enc_list[k].weight)
            assert torch.allclose(sae.weights_dec[k].weight, dec_list[k].weight)

    test_ae()
    test_sae()
    


def test_sbae_soae_projection_properties():
    """
    This is to verify that SBAE and SOAE are "representation-consistent", i.e.:
    E(D(c)) = c for any suitable c (up to some tolerance).
    """

    # Config
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    np.random.seed(1)
    torch.manual_seed(1)

    # Synthetic data
    ns = 1000
    nh = 256
    red_dims = (nh, 40, 20, 3)
    lat_distr = torch.randn((ns, red_dims[-1]), device = device)
    theta = 0.1
    bilipactivation = HypAct(theta)

    # Initialize architectures
    soae = SOAE(red_dims, bilipactivation)
    soae.standard()
    soae.to(device)
    sbae = SBAE(red_dims, bilipactivation)
    sbae.standard()
    sbae.to(device)

    # Test E(D(c)) == c
    for model in soae, sbae:
        diff = model.encode(model.decode(lat_distr)) - lat_distr
        cond = np.allclose(diff.pow(2).sum(axis = 1).mean().item(), 0)
        assert cond
