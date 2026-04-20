import os

import numpy as np 
import torch
from dlroms import * 
from dolfin import *

from blocks import Normalizer
from NestedPOD import NestedPOD
from modules import SOAE, SBAE
from utils import generate_data_for_tests_suite
from activations import HypAct
from modules import SBAE, SOAE
import training



def test_sbae_soae_orth_after_training():
    """
    We use this test to verify that (bi-)orthogonality is preserved after 
    training SBAE and SOAEs. This is done both for the EYS and the standard 
    random initialization.
    """

    # Config
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    np.random.seed(1)
    torch.manual_seed(1)

    # Generate data
    utrain, uval, utest = generate_data_for_tests_suite(
        device = device, ns = 200
    )

    # Instantiate normalizer
    normalizer = Normalizer(utrain = utrain)

    # Instantiate some activation
    theta = 0.3
    bilipactivation = HypAct(theta) 
    bilipactivation.setup()
    red_dims = (utrain.shape[1], 20, 10, 3)
    nested_pod = NestedPOD(
        normalizer.forward(utrain), 
        red_dims, 
        bilipactivation
    )

    for init in ('eys', 'standard'):
    
        # SOAE
        soae = SOAE(red_dims, bilipactivation)
        if init == 'eys':
            soae.eys(nested_pod = nested_pod)
        elif init == 'standard': 
            soae.standard()
        else:
            raise ValueError()
        soae.to(device)
        
        # SBAE
        sbae = SBAE(red_dims, bilipactivation)
        if init == 'eys':
            sbae.eys(nested_pod = nested_pod)
        elif init == 'standard': 
            sbae.standard()
        else:
            raise ValueError()
        sbae.to(device)
        
        # Iterating over models (SBAE, SOAE)
        for model, name in zip((sbae, soae), ('SBAE', 'SOAE')):

            # Training
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
            _, _ = training.train(
                U = torch.utils.data.DataLoader(
                        normalizer.forward(utrain), 
                        batch_size = 8, 
                        shuffle = True
                    ),
                model = model,
                optimizer = optimizer, 
                U_val = normalizer.forward(uval),
                patience = 10,
                epochs = 10,
                monitor_history = True,
                verbose = False,
                ckpt_path = os.path.join('tests', name + '_tests.pt')
            )

            # Assert orthogonality in SBAE and SOAE at each level
            for level in model.weights: 
                if name == 'SBAE':
                    E = level.enc_mat
                    D = level.dec_mat
                    I_appx = E @ D
                else:   
                    V = level.weight.data
                    I_appx = V @ V.T
                err = (torch.eye(I_appx.shape[0]).to(device) - I_appx)**2
                assert np.allclose(torch.sum(err).item(), 0)

            # Assert projection properties
            utest_norm = normalizer.forward(utest)
            for model in soae, sbae:
                diff = model.encode(model(utest_norm))-model.encode(utest_norm)
                cond = np.allclose(diff.pow(2).sum(axis = 1).mean().item(), 0)
                assert cond






