import numpy as np
import torch
import os
import time
import random
from scipy.optimize import bisect
from dlroms import msei, mrei, euclidean

from modules import AE, SAE, SBAE, SOAE
from utils import loadexp, save_analysis
from blocks import Normalizer
import training
import metrics
from activations import LeakyReLU
from NestedPOD import NestedPOD


#-------------------------------------------------------------------------------
# Setup
#-------------------------------------------------------------------------------

# General setup
seed = 1
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
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
def get_experimental_setup(ns_train):
    epochs = int(np.ceil(200 * 1500 / ns_train))
    lr = 1e-3
    batch_size = 8
    patience = epochs
    exp_setup = (epochs, lr, batch_size, patience)
    print('- Training for %d epochs, with lr = %1.2e, batch size = %d, patience = %d'\
        % exp_setup)
    return exp_setup


# savepath setup 
savepath = os.path.join('..', 'results')
if not os.path.exists(savepath):
    os.makedirs(savepath)
ckpt_dir = os.path.join(savepath, 'ckpts', 'sample_size_robustness_ckpts')
os.makedirs(ckpt_dir, exist_ok = True)
analysis_dir = os.path.join(savepath, 'analyses')
os.makedirs(analysis_dir, exist_ok = True)
savedir = os.path.join(
    analysis_dir, 'sample_size_robustness_analysis.obj'
)
print('- The analysis results will be saved to: %s' % savedir)
print('- The ckpts will be saved to: %s' % ckpt_dir)
print('-' * 128)


# activation setup
sharpness = 3
alphaLeakyRelu = lambda alpha: LeakyReLU(alpha, 1.25)
alpha_leaky = bisect(
    lambda alpha: (alphaLeakyRelu(alpha).sharpness - sharpness), 
    a = 1.3,
    b = 15
)
print(alpha_leaky)

# loadpath setup  
datapath = os.path.join('..', 'data')
def get_exp_rod(data_id):
    meshpath = os.path.join(datapath, 'rod_mesh.xml')
    trainval_datapath = os.path.join(
        datapath, 
        'rod_snapshots_sample_size_robustness_trainval_id' \
            + str(data_id) + '.npz'
    )
    test_datapath = os.path.join(
        datapath, 
        'rod_snapshots_sample_size_robustness_test.npz'
    )
    hidden_dims = (15, 7, 5)
    exp_rod = {
        'meshpath' : meshpath, 
        'trainval_datapath' : trainval_datapath, 
        'test_datapath' : test_datapath, 
        'hidden_dims' : hidden_dims
    }
    return exp_rod

# Architecture setup
def loadexpwrapper(meshpath, datapath, hidden_dims, split):
    data_split, mesh = loadexp(meshpath, datapath, split, device)
    data_split_no_none = [elem for elem in data_split if elem is not None]
    n0 = data_split_no_none[0].shape[1]
    red_dims = (n0, ) + hidden_dims
    return data_split, mesh, red_dims


# Splitting setup
train_perc_ontrainval = 0.8
val_perc_ontrainval = 0.2
ns_trainval_range = 50 * 3**np.arange(4)
print('N_train + N_val = ', ns_trainval_range)


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
analysis['data_id'] = list()
analysis['ns_train'] = list()
analysis['ns_val'] = list()
analysis['ns_trainval'] = list()


for data_id in range(1,6):
    for ns_trainval in ns_trainval_range:

        # Load trainval data
        train_split = int(train_perc_ontrainval * ns_trainval)
        val_split = int(val_perc_ontrainval * ns_trainval)
        trainval_split = [train_split, val_split, 0]
        test_split = [0, 0, 500]
        exp_rod = get_exp_rod(data_id = data_id)
        trainval_data_split, mesh, red_dims = loadexpwrapper(
            meshpath = exp_rod['meshpath'],
            datapath = exp_rod['trainval_datapath'],
            hidden_dims = exp_rod['hidden_dims'],
            split = trainval_split
        )
        test_data_split, mesh, red_dims = loadexpwrapper(
            meshpath = exp_rod['meshpath'],
            datapath = exp_rod['test_datapath'],
            hidden_dims = exp_rod['hidden_dims'],
            split = test_split
        )
        (utrain, uval, _) = trainval_data_split
        (_, _, utest) = test_data_split
        normalizer = Normalizer(utrain)


        # Define experimental setup
        epochs, lr, batch_size, patience = get_experimental_setup(
            ns_train = train_split
        )
        

        # Activation function instatiation
        bilipactivation = alphaLeakyRelu(alpha_leaky)
        bilipactivation.setup()

        for initializer in ('eys', 'standard'):

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

            # Iterate over models
            for model_name, model in model_configs:
                model_full_string = 'rod_' + model_name + '_' + \
                    initializer + '_leaky_sharp=' + str(sharpness) + \
                    '_train' + str(train_split) + '_val' + str(val_split) + \
                    '_dataid' + str(data_id) + '.pt'
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
                test_pred = normalizer.backward(
                    model(normalizer.forward(utest))
                )
                elapsed_time = time.time() - t0


                # Save analysis
                analysis['experiment'].append('rod')
                analysis['act_name'].append('leaky')
                analysis['act_parameter'].append(alpha_leaky)
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
                analysis['data_id'].append(data_id)
                analysis['ns_train'].append(train_split)
                analysis['ns_val'].append(val_split)
                analysis['ns_trainval'].append(ns_trainval)
                save_analysis(
                    analysis_dict = analysis,
                    filename = savedir
                )        
            print('-' * 128)
    