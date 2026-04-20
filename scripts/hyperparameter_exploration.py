import os
import random
import warnings

from ray import tune
import ray
import torch
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
import numpy as np
from scipy.optimize import bisect

from modules import SAE
from utils import loadexp, save_analysis
from blocks import Normalizer
import training
import metrics
from activations import LeakyReLU, HypAct
from NestedPOD import NestedPOD

os.environ['TQDM_DISABLE'] = 'True'


#------------------------------------------------------------------------------#
# Functions definition
#------------------------------------------------------------------------------#


# Define common train function ------------------------------------------------#
def train_hyp_opt(
    config,
    red_dim_options,
    normalizer, 
    utrain,
    uval,
    ckpt_dir,
    train_data_loader,
    scheduler_options
):
    
    # Reproducibility setting
    raytune_seed = int(config['seed'])
    random.seed(raytune_seed)
    np.random.seed(raytune_seed)
    torch.manual_seed(raytune_seed)

    # Activation setup
    red_dims = red_dim_options[config['red_dims_cfg']]
    bilipactivation = activations_options[config['act_cfg']]
    bilipactivation.setup()

    # Initialize architecture
    nested_pod = NestedPOD(
        snapshots = normalizer.forward(utrain), 
        red_dims = red_dims, 
        bilipactivation = bilipactivation
    )
    sae = SAE(red_dims, bilipactivation)
    if config['init_cfg'] == 'eys':
        sae.eys(nested_pod = nested_pod)
    elif config['init_cfg'] == 'standard':
        sae.standard()
    else:
        raise ValueError()
    sae.to(device)

    # Training
    optimizer = torch.optim.Adam(sae.parameters(), lr = config['lr_cfg'])
    os.makedirs(ckpt_dir, exist_ok = True)
    ckpt_path = os.path.join(ckpt_dir, str(config) + '.pt')
    sae, history = training.train(
        U = train_data_loader,
        model = sae,
        optimizer = optimizer, 
        U_val = normalizer.forward(uval),
        patience = patience,
        scheduler = scheduler_options[config['scheduler_cfg']](optimizer),
        epochs = epochs,
        monitor_history = True,
        verbose = False,
        ckpt_path = ckpt_path
    )

    # Report validation loss
    val_pred = normalizer.backward(
        sae(normalizer.forward(uval))
    )
    tune.report(
        {'loss' : metrics.mse(utrue = uval, upred = val_pred).item(),
         'last_epoch' : len(history['loss_train'])}
    )


# Helper utility function -----------------------------------------------------#
def tuple_to_dict(tuple_in):
    dict_out = dict()
    for elem in tuple_in:
        dict_out[str(elem)] = elem
    return dict_out


# Config setup helper ---------------------------------------------------------#
def get_config(train_data_loader, red_dim_options):

    # Define scheduler options
    onecycle_scheduler = lambda optimizer: OneCycleLR(
        optimizer, 
        max_lr = 1e-3,  
        epochs = epochs,
        steps_per_epoch = len(train_data_loader)
    )
    cosineann_scheduler = lambda optimizer: CosineAnnealingLR(
        optimizer, 
        T_max = epochs
    )
    scheduler_options = {
        'no' : lambda optimizer : None,
        'onecycle' : onecycle_scheduler,
        'cosineann' : cosineann_scheduler,
    }

    # Set config for hyperparameter optimization
    config = {
        'red_dims_cfg' : tune.grid_search(list(red_dim_options.keys())),
        'scheduler_cfg' : tune.grid_search(list(scheduler_options.keys())),
        'init_cfg' : tune.grid_search(['eys', 'standard']),
        'lr_cfg' : tune.grid_search([1e-3, 6e-4, 3e-4]),
        'act_cfg' : tune.grid_search(list(activations_options.keys())),
        'seed' : tune.randint(0, int(1e9))
    }

    return config, scheduler_options


# Load data utility -----------------------------------------------------------#
def load_data_helper(exp):
    (utrain, uval, _), _ = loadexp(*exp, split = [200, 100, 0])
    utrain = utrain.to(device)
    uval = uval.to(device)
    normalizer = Normalizer(utrain)
    train_data_loader = torch.utils.data.DataLoader(
        normalizer.forward(utrain), 
        batch_size = batch_size, 
        shuffle = True
    )
    return (utrain, uval), train_data_loader, normalizer


#------------------------------------------------------------------------------#
# PGA
#------------------------------------------------------------------------------#

def run_hypopt_pga(tune_config):
    
    # Load and process data
    (utrain, uval), train_data_loader, normalizer = load_data_helper(
        exp = exp_gaussian
    )

    # Set checkpoint directory
    ckpt_dir = os.path.join(
        os.getcwd(), resultspath, 'ckpts', 'hypopt_ckpts', 'pga'
    )

    # Set architecture options
    red_dim_options = tuple_to_dict(
        tuple_in = (
            [514, 64, 3],
            [514, 64, 15, 3],
            [514, 64, 15, 7, 3],
            [514, 32, 15, 3],
            [514, 16, 10, 3],
            [514, 64, 15, 5],
            [514, 64, 18, 7]
        )
    )

    # Get config
    config, scheduler_options= get_config(
        train_data_loader = train_data_loader,
        red_dim_options = red_dim_options
    )
    
    # Setup hypopt function
    pga_hypopt_fn = lambda config: train_hyp_opt(
        config, 
        red_dim_options = red_dim_options,
        normalizer = normalizer,
        utrain = utrain,
        uval = uval,
        ckpt_dir = ckpt_dir,
        train_data_loader = train_data_loader,
        scheduler_options = scheduler_options
    )

    # Tune
    tuner = tune.Tuner(
        tune.with_resources(
            pga_hypopt_fn,
            resources = {'cpu' : 1}
        ),
        param_space = config,
        tune_config = tune_config,
        run_config = tune.RunConfig(
            storage_path = os.path.join(
                os.getcwd(), resultspath, 'raytune', 'pga'
            )
        )
    )
    results = tuner.fit()

    return results



#------------------------------------------------------------------------------#
# ROD
#------------------------------------------------------------------------------#

def run_hypopt_rod(tune_config):
    
    # Load and process data
    (utrain, uval), train_data_loader, normalizer = load_data_helper(
        exp = exp_rod
    )

    # Set checkpoint directory
    ckpt_dir = os.path.join(
        os.getcwd(), resultspath, 'ckpts', 'hypopt_ckpts', 'rod'
    )

    # Set architecture options
    red_dim_options = tuple_to_dict(
        tuple_in = (
            [4347, 15, 5],
            [4347, 15, 7, 5],
            [4347, 20, 7],
            [4347, 30, 20, 10, 5],
            [4347, 30, 10, 5],
            [4347, 20, 10, 7],
            [4347, 30, 20, 15, 10, 5]
        )
    )

    config, scheduler_options= get_config(
        train_data_loader = train_data_loader,
        red_dim_options = red_dim_options
    )

    # Setup hypopt function
    rod_hypopt_fn = lambda config: train_hyp_opt(
        config, 
        red_dim_options = red_dim_options,
        normalizer = normalizer,
        utrain = utrain,
        uval = uval,
        ckpt_dir = ckpt_dir,
        train_data_loader = train_data_loader,
        scheduler_options = scheduler_options
    )

    # Tune
    tuner = tune.Tuner(
        tune.with_resources(
            rod_hypopt_fn,
            resources = {'cpu' : 1}
        ),
        param_space = config,
        tune_config = tune_config,
        run_config = tune.RunConfig(
            storage_path = os.path.join(
                os.getcwd(), resultspath, 'raytune', 'rod'
            )
        )
    )
    results = tuner.fit()

    return results


#------------------------------------------------------------------------------#
# ELS
#------------------------------------------------------------------------------#

def run_hypopt_els(tune_config):
    
    # Load and process data
    (utrain, uval), train_data_loader, normalizer = load_data_helper(
        exp = exp_shield
    )

    # Set checkpoint directory
    ckpt_dir = os.path.join(
        os.getcwd(), '..', 'results_hypexp', 'ckpts', 'hypopt_ckpts', 'els'
    )

    # Set architecture options
    red_dim_options = tuple_to_dict(
        tuple_in = (
            [882, 8, 5],
            [882, 15, 10, 8],
            [882, 15, 10],
            [882, 10, 8],
            [882, 30, 15, 10, 8],
            [882, 30, 10],
            [882, 15, 9, 6]
        )
    )

    # Get config
    config, scheduler_options = get_config(
        train_data_loader = train_data_loader,
        red_dim_options = red_dim_options
    )

    # Setup hypopt function
    els_hypopt_fn = lambda config: train_hyp_opt(
        config, 
        red_dim_options = red_dim_options,
        normalizer = normalizer,
        utrain = utrain,
        uval = uval,
        ckpt_dir = ckpt_dir,
        train_data_loader = train_data_loader,
        scheduler_options = scheduler_options
    )

    # Tune
    tuner = tune.Tuner(
        tune.with_resources(
            els_hypopt_fn,
            resources = {'cpu' : 1}
        ),
        param_space = config,
        tune_config = tune_config,
        run_config = tune.RunConfig(
            storage_path = os.path.join(
                os.getcwd(), resultspath, 'raytune', 'els'
            )
        )
    )
    
    results = tuner.fit()

    return results



#------------------------------------------------------------------------------#
# Main program
#------------------------------------------------------------------------------#

if __name__ == '__main__':

    # Directories -------------------------------------------------------------#
    
    # Results and analysis
    resultspath = os.path.join('..', 'results')
    try:
        os.makedirs(resultspath)
    except:
        warnings.warn(
            'resultspath already exists; may overwrite previous results.'
        )
    if not os.path.exists(os.path.join(resultspath, 'analyses')):
        os.makedirs(os.path.join(resultspath, 'analyses'))

    # Data
    datapath = os.path.join('..', 'data')
    exp_rod = (
        os.path.join(datapath, 'rod_mesh.xml'), 
        os.path.join(datapath,'rod_snapshots.npz')
    )
    exp_shield = (
        os.path.join(datapath, 'shield_mesh.xml'), 
        os.path.join(datapath,'shield_snapshots.npz')
    )
    exp_gaussian = (
        os.path.join(datapath, 'gaussian_mesh.xml'), 
        os.path.join(datapath,'gaussian_snapshots.npz')
    )



    # General setup -----------------------------------------------------------#
    device = 'cpu'
    seed = 1
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


    # Activations -------------------------------------------------------------#
    sharpnesses = (0.5, 3)
    aLeakyRelu = lambda alpha: LeakyReLU(alpha, 1.25)
    alpha_leaky = [
        bisect(
            lambda alpha: (aLeakyRelu(alpha).sharpness - sharpness), 
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
    activations_options = {
        'LeakyReLU(%1.4f, 1.25)' % alpha_leaky[0] : aLeakyRelu(alpha_leaky[0]),
        'LeakyReLU(%1.4f, 1.25)' % alpha_leaky[1] : aLeakyRelu(alpha_leaky[1]),
        'Hypact(%1.4f)' % theta_hypact[0] : HypAct(theta_hypact[0]),
        'HypAct(%1.4f)' % theta_hypact[1] : HypAct(theta_hypact[1])
    }


    # Experimental setup ------------------------------------------------------#
    patience = 500
    epochs = 1500
    batch_size = 8
    tune_config = tune.TuneConfig(
        num_samples = 3, 
        max_concurrent_trials = 60 
    )


    # Ray setup ---------------------------------------------------------------#
    if ray.is_initialized():
        ray.shutdown()
    ray.init()


    # Run and save ------------------------------------------------------------#
    results_pga = run_hypopt_pga(tune_config = tune_config)
    save_analysis(
        analysis_dict = results_pga, 
        filename = os.path.join(resultspath, 'analyses', 'hypexp_pga.obj')
    )
    results_rod = run_hypopt_rod(tune_config = tune_config)
    save_analysis(
        analysis_dict = results_rod, 
        filename = os.path.join(resultspath, 'analyses', 'hypexp_rod.obj')
    )
    results_els = run_hypopt_els(tune_config = tune_config)
    save_analysis(
        analysis_dict = results_els, 
        filename = os.path.join(resultspath, 'analyses', 'hypexp_els.obj')
    )