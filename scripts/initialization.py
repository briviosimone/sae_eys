import numpy as np
import tqdm
import torch

import os
import random
from blocks import Normalizer
from modules import SAE, SOAE
from activations import LeakyReLU, HypAct, BilipActivation
from utils import loadexp, save_analysis
from NestedPOD import NestedPOD
import metrics


#------------------------------------------------------------------------------#
# Functions definition
#------------------------------------------------------------------------------#

def compute_mse_val(
    model_class,
    red_dims : tuple,
    bilipactivation : BilipActivation
):
    # Random trials for standard random initialization
    curr_mses_standard = list()
    for _ in range(n_trials):
        model_standard = model_class(red_dims, bilipactivation)
        model_standard.standard()
        model_standard.to(device)
        upred_standard = normalizer.backward(
            model_standard(normalizer.forward(uval))
        )
        curr_mses_standard.append(
            metrics.mse(utrue = uval, upred = upred_standard).item()
        )
    # EYS initialization
    nested_pod = NestedPOD(
        normalizer.forward(utrain), red_dims, bilipactivation
    )
    model_eys = model_class(red_dims, bilipactivation)
    model_eys.eys(nested_pod = nested_pod)
    model_eys.to(device)
    upred_eys = normalizer.backward(model_eys(normalizer.forward(uval)))
    curr_mse_eys = metrics.mse(utrue = uval, upred = upred_eys).item()

    return curr_mses_standard, curr_mse_eys





def analyze_latent_dim(
    model_class, 
    activation_class, 
    act_param_range : np.array,
    latent_dims : np.array
):

    # Initialize collections for the analysis
    init_analysis_latent = dict()
    init_analysis_latent['mse_eys'] = list()
    init_analysis_latent['mse_standard'] = list()
    init_analysis_latent['sharpnesses'] = list()
    init_analysis_latent['latent_dim'] = list()

    for latent_dim in tqdm.tqdm(latent_dims):
        for act_param in act_param_range:
            # Activation function setup
            bilipactivation = activation_class(act_param) 
            bilipactivation.setup()
            # Skeleton setup
            red_dims = (utrain.shape[1], 20, latent_dim)
            # Compute validation MSEs for {standard, eys} inits
            curr_mses_standard, curr_mse_eys = compute_mse_val(
                model_class = model_class,
                red_dims = red_dims,
                bilipactivation = bilipactivation
            )
            # Collect quantities for analysis
            init_analysis_latent['mse_eys'].append(curr_mse_eys)
            init_analysis_latent['mse_standard'].append(
                np.min(curr_mses_standard)
            )
            init_analysis_latent['sharpnesses'].append(
                bilipactivation.sharpness
            )
            init_analysis_latent['latent_dim'].append(latent_dim)
    
    return init_analysis_latent




def analyze_depth(
    model_class, 
    activation_class, 
    act_param_range : np.array,
    red_dims_collection : tuple[list]
):

    # Initialize collections for the analysis
    init_analysis_depth = dict()
    init_analysis_depth['mse_eys'] = list()
    init_analysis_depth ['mse_standard'] = list()
    init_analysis_depth['sharpnesses'] = list()
    init_analysis_depth['depth'] = list()

    for red_dims in tqdm.tqdm(red_dims_collection):
        for act_param in act_param_range:
            # Activation function setup
            bilipactivation = activation_class(act_param) 
            bilipactivation.setup()
            # Skeleton setup
            curr_mses_standard = list()
            # Compute validation MSEs for {standard, eys} inits
            curr_mses_standard, curr_mse_eys = compute_mse_val(
                model_class = model_class,
                red_dims = red_dims,
                bilipactivation = bilipactivation
            )
            # Collect quantities for analysis
            init_analysis_depth['mse_eys'].append(curr_mse_eys)
            init_analysis_depth['mse_standard'].append(
                np.min(curr_mses_standard)
            )
            init_analysis_depth['sharpnesses'].append(bilipactivation.sharpness)
            init_analysis_depth['depth'].append(len(red_dims) - 1)

    return init_analysis_depth



#------------------------------------------------------------------------------#
# Main program
#------------------------------------------------------------------------------#

if __name__ == '__main__':

    # Directories -------------------------------------------------------------#

    # Upload data
    datapath = os.path.join('..', 'data')
    exp_gaussian = (
        os.path.join(datapath, 'gaussian_mesh.xml'), 
        os.path.join(datapath,'gaussian_snapshots.npz')
    )

    # Set savepath
    savepath = os.path.join('..', 'results')
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    analysis_path = os.path.join(savepath, 'analyses')
    if not os.path.exists(analysis_path):
        os.makedirs(analysis_path)
    savedir = os.path.join(analysis_path, 'init_study.obj')

    # General setup -----------------------------------------------------------#
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


    # Experimental setup ------------------------------------------------------#

    # Select test case
    (utrain, uval, _), mesh = loadexp(*exp_gaussian, split = [200, 100, 0])
    normalizer = Normalizer(utrain = utrain)
    n0 = utrain.shape[1]

    # Select activations and range for the activation function parameter
    activation_classes = (
        HypAct,
        lambda alpha: LeakyReLU(alpha, 5/4)
    )
    thetas = np.linspace(0.1, np.pi / 6, 7) # for Hypact_{\theta}
    alphas = np.linspace(5/6, 7, 7) # for LeakyRelu_{alpha,5/4}

    # Select number of random trials
    n_trials = 100

    # Select latent dimension range
    latent_dims = np.arange(1,16)

    # Collection of skeletons corresponding to increasing depths
    red_dims_collection = (
        [n0, 65, 3],
        [n0, 65, 5, 3],
        [n0, 65, 9, 5, 3],
        [n0, 65, 17, 9, 5, 3],
        [n0, 65, 33, 17, 9, 5, 3]
    )


    # Experiments -------------------------------------------------------------#

    init_study = dict()
    for model_class in (SAE, SOAE):
        for activation_class, act_param_range, act_name in zip(
            activation_classes, (thetas, alphas), ('hypact', 'leaky')
        ):
            curr_id =  model_class.__name__ + '_' + act_name
            curr_id = curr_id.lower()
            init_analysis_latent = analyze_latent_dim(
                model_class = model_class, 
                activation_class = activation_class, 
                act_param_range = act_param_range,
                latent_dims = latent_dims
            )
            init_analysis_depth = analyze_depth(
                model_class = model_class, 
                activation_class = activation_class, 
                act_param_range = act_param_range,
                red_dims_collection = red_dims_collection
            )
            init_study[curr_id] = dict()
            init_study[curr_id]['latent_dim'] = init_analysis_latent
            init_study[curr_id]['depth'] = init_analysis_depth
            save_analysis(
                analysis_dict = init_study,
                filename = savedir
            )   