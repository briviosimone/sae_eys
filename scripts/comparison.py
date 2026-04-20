import numpy as np
import torch
import os
import time
import random
from scipy.optimize import bisect


from modules import AE, SAE, SBAE, SOAE
from utils import loadexp, save_analysis
from blocks import Normalizer
import training
import metrics
from activations import LeakyReLU, HypAct, BilipActivationConfig
from NestedPOD import NestedPOD


#-------------------------------------------------------------------------------
# Setup
#-------------------------------------------------------------------------------

# General setup
seed = 1
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('-' * 128)
print('- Using device %s with seed %d' % (device, seed))
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    print('Setting reproducibility setup for GPU device %s' % device)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

# experimental setup 
epochs = 1500
lr = 1e-3
batch_size = 8
patience = 500
print('- Training for %d epochs, with lr = %1.2e, batch size = %d, patience = %d'\
      % (epochs, lr, batch_size, patience))

# savepath setup 
savepath = os.path.join('..', 'results')
if not os.path.exists(savepath):
    os.makedirs(savepath)
ckpt_dir = os.path.join(savepath, 'ckpts', 'comparison_ckpts')
os.makedirs(ckpt_dir, exist_ok = True)
analysis_dir = os.path.join(savepath, 'analyses')
os.makedirs(analysis_dir, exist_ok = True)
savedir = os.path.join(analysis_dir, 'comparison_analysis.obj')
print('- The analysis results will be saved to: %s' % savedir)
print('- The ckpts will be saved to: %s' % ckpt_dir)
print('-' * 128)

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
datapath = os.path.join('..', 'data')
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
    data_split, mesh = loadexp(meshpath, datapath, split = [200, 100, 500])
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
analysis['mse_5%_ci'] = list()
analysis['mre'] = list()
analysis['mre_5%_ci'] = list()
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
            
                print('Running with %s init., using %s activation with activ. param = %1.2e, corresponding to sharpness = %1.2e' \
                      % (initializer, activation_config.name, parameter, sharpness))
                
                # Activation function instantiation
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
                else:
                    raise ValueError(f'Unavailable {initializer} initializer.')

                
                # iterate over models
                for model_name, model in model_configs:
                    model_full_string = name_exp + '_' + model_name + '_' + \
                        initializer + '_' + activation_config.name + \
                        '_sharp=' + str(sharpness) + '.pt'
                    ckpt_path = os.path.join(ckpt_dir, model_full_string)
                    print('-> ' + model_name)
                    model.to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
                    t0 = time.time()
                    _, history_monitor = training.train(
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
                        ckpt_path = ckpt_path
                    )
                    elapsed_time = time.time() - t0

                    test_pred = normalizer.backward(
                        model(normalizer.forward(utest))
                    )

                    # Save analysis
                    analysis['experiment'].append(name_exp)
                    analysis['act_name'].append(activation_config.name)
                    analysis['act_parameter'].append(parameter)
                    analysis['act_sharpness'].append(sharpness)
                    analysis['initialization'].append(initializer)
                    analysis['model_name'].append(model_name)
                    analysis['mse'].append(
                        metrics.mse(utrue = utest, upred = test_pred).item()
                    )
                    analysis['mse_5%_ci'].append(
                        metrics.band95_squared_error(
                            utrue = utest, upred = test_pred
                        ).item()
                    )
                    analysis['mre'].append(
                        metrics.mre(utrue = utest, upred = test_pred).item()
                    )
                    analysis['mre_5%_ci'].append(
                        metrics.band95_relative_error(
                            utrue = utest, upred = test_pred
                        ).item()
                    )
                    analysis['elapsed_time'].append(elapsed_time)
                    analysis['epochs'].append(len(history_monitor['loss_val']))
                    analysis['monitors'].append(history_monitor)
                    save_analysis(
                        analysis_dict = analysis,
                        filename = savedir
                    )        
                print('-' * 128)           