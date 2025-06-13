from dlroms import *
import numpy as np
import torch
from src.modules import *
from src.utils import *
from src import training
from src.activations import *
from src.NestedPOD import NestedPOD
from scipy.optimize import bisect
import os
import time


#-------------------------------------------------------------------------------
# Setup
#-------------------------------------------------------------------------------

# General setup
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
np.random.seed(1)
torch.manual_seed(1)

# experimental setup 
epochs = 1500
lr = 1e-3
batch_size = 8
patience = 500

# savepath setup 
savepath = 'results'
if not os.path.exists(savepath):
    os.makedirs(savepath)
savedir = os.path.join(savepath, 'comparison_analysis.obj')


# activations setup
sharpnesses = (0.5, 3)
alphaLeakyRelu = lambda alpha: LeakyReLU(alpha, 1.25)
alpha_leaky = [
    bisect(
        lambda alpha: (alphaLeakyRelu(alpha).sharpness - sharpness), 
        a = a,
        b = b
    )
    for (sharpness, (a,b)) in zip(sharpnesses, ((0.01,1.2), (1.3, 15)))
]
theta_hypact = [
    bisect(
        lambda theta: (HypAct(theta).sharpness - sharpness), 
        a = 0.01,
        b = 0.77
    )
    for sharpness in sharpnesses
]
activation_configs = (
    BilipActivationConfig(
        name = 'hypact', 
        bilipactivation = HypAct,
        parameters = theta_hypact,
        sharpnesses = sharpnesses
    ),
    BilipActivationConfig(
        name = 'leaky', 
        bilipactivation = alphaLeakyRelu,
        parameters = alpha_leaky,
        sharpnesses = sharpnesses
    ),
)


# loadpath setup  
datapath = 'data'
exp_rod = (
    os.path.join(datapath, 'rod_mesh.xml'), 
    os.path.join(datapath,'rod_snapshots.npz'),
    (15, 7, 5)
)
exp_shield = (
    os.path.join(datapath, 'shield_mesh.xml'), 
    os.path.join(datapath,'shield_snapshots.npz'),
    (15, 10, 8)
)
exp_gaussian = (
    os.path.join(datapath, 'gaussian_mesh.xml'), 
    os.path.join(datapath,'gaussian_snapshots.npz'),
    (64, 15, 3)
)
experiments = zip(
    ('rod','shield','gaussian'),
    (exp_rod, exp_shield, exp_gaussian)
)


# Architecture setup
def loadexpwrapper(meshpath, datapath, hidden_dims):
    data_split, mesh = loadexp(meshpath, datapath)
    n0 = data_split[0].shape[1]
    red_dims = (n0, ) + hidden_dims
    return data_split, mesh, red_dims




#-------------------------------------------------------------------------------
# Analysis
#-------------------------------------------------------------------------------

analysis = dict()
analysis['experiment'] = list()
analysis['act_name'] = list()
analysis['act_parameter'] = list()
analysis['act_sharpness'] = list()
analysis['initialization'] = list()
analysis['model_name'] = list()
analysis['mse'] = list()
analysis['mre'] = list()
analysis['elapsed_time'] = list()
analysis['epochs'] = list()
analysis['monitors'] = list()

for name_exp, exp in experiments:
    
    # Load experiment
    data_split, mesh, red_dims = loadexpwrapper(*exp)
    (utrain, uval, utest) = data_split
    normalizer = Normalizer(utrain)
    
    for activation_config in activation_configs:
        param_and_sharpness = zip(
            activation_config.parameters,
            activation_config.sharpnesses
        )
        for parameter, sharpness in param_and_sharpness:
            for initializer in ('standard', 'eys'):
            
                print(initializer, parameter, sharpness)
                bilipactivation = activation_config.bilipactivation(parameter)
                bilipactivation.setup()
                

                # Generate models
                ae = AE(red_dims, bilipactivation)
                sae = SAE(red_dims, bilipactivation)
                sbae = SBAE(red_dims, bilipactivation)
                soae = SOAE(red_dims, bilipactivation)

                # Model configs
                model_configs = zip(
                    ('ae', 'sae', 'sbae', 'soae'), 
                    (ae, sae, sbae, soae)
                )

                # Model initialization
                if initializer == 'standard':
                    ae.standard()
                    sae.standard()
                    sbae.standard()
                    soae.standard()
                elif initializer == 'eys':
                    nested_pod = NestedPOD(
                        snapshots = normalizer.forward(utrain), 
                        red_dims = red_dims, 
                        bilipactivation = bilipactivation
                    )
                    ae.eys(nested_pod = nested_pod)
                    sae.eys(nested_pod = nested_pod)
                    sbae.eys(nested_pod = nested_pod)
                    soae.eys(nested_pod = nested_pod)

                
                # iterate over models
                for model_name, model in model_configs:

                    model.to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
                    t0 = time.time()
                    _, history_monitor = training.train(
                        J = training.classic_ae_mse,
                        U = torch.utils.data.DataLoader(
                                normalizer.forward(utrain), 
                                batch_size = batch_size, 
                                shuffle = True
                            ),
                        model = model,
                        optimizer = optimizer, 
                        U_val = normalizer.forward(uval),
                        patience = patience,
                        epochs = epochs,
                        monitor_history = True,
                        verbose = False,
                        model_name = 'comparison'
                    )
                    test_pred = normalizer.backward(
                        model(normalizer.forward(utest))
                    )
                    elapsed_time = time.time() - t0

                    # Save analysis
                    analysis['experiment'].append(name_exp)
                    analysis['act_name'].append(activation_config.name)
                    analysis['act_parameter'].append(parameter)
                    analysis['act_sharpness'].append(sharpness)
                    analysis['initialization'].append(initializer)
                    analysis['model_name'].append(model_name)
                    analysis['mse'].append(msei(euclidean)(utest, test_pred))
                    analysis['mre'].append(mrei(euclidean)(utest, test_pred))
                    analysis['elapsed_time'].append(elapsed_time)
                    analysis['epochs'].append(len(history_monitor['loss_val']))
                    analysis['monitors'].append(history_monitor)
                    save_analysis(
                        analysis_dict = analysis,
                        filename = savedir
                    )                   