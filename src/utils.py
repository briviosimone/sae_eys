import pickle
from typing import Optional
from collections.abc import Sequence

from dlroms import dv, fe
import numpy as np
import torch



def split_data(
    snapshots : torch.tensor,
    split : list[int]
):
    """ Splits data in (train, val, test).

    Args:
        snapshots (torch.tensor): contains all the snapshots 
                                  (nsamples-by-ndofs format).
        split (list[int]): how many samples for (train, val, test).

    Returns:
        split data (train, val, test)
    """
    # Print data info
    print('All dataset samples = %d' % snapshots.shape[0])
    ntrain, nval, ntest = split

    # Split data
    if ntrain > 0:
        utrain = snapshots[:ntrain]
        print('Train samples idxs -> [%d,%d]' % (0,ntrain-1))
    else:
        utrain = None
        print('No train samples: returning utrain = None.')
    if nval > 0:
        uval = snapshots[ntrain:ntrain+nval]
        print('Val   samples idxs -> [%d,%d]' % (ntrain, ntrain+nval-1))
    else:
        uval = None
        print('No val samples: returning uval = None.')
    if ntest > 0:
        utest = snapshots[-ntest:]
        print('Test samples idxs  -> [%d,%d]' % \
              (snapshots.shape[0]-ntest, snapshots.shape[0]-1))
    else:
        utest = None
        print('No test samples: returning utest = None.')

    # Checks any test overlapping
    if (ntrain + nval) > snapshots.shape[0] - ntest:
        raise ValueError('Test set is overlapping with train and val sets')
    data_split = (utrain, uval, utest)

    return data_split


def loadexp(
    meshpath : str, 
    datapath : str,
    split : list[int],
    device : str = None
):
    """ Loads mesh and data given their paths.

    Args:
        meshpath (str): path/to/mesh.
        datapath (str): path/to/data.
        split (list[float]): how many samples for (train, val, test).
        device (str): name of the device to load data to.
    
    Returns:
        split data (train, val, test), and the mesh.
    """

    # Read mesh and data
    print('-' * 128)
    print('Reading mesh from: %s' % meshpath)
    print('Reading data from: %s' % datapath)
    mesh = fe.loadmesh(meshpath)
    data = np.load(datapath)
    mu, u = data['mu'], data['u']

    # Move data to device 
    mu, u = dv.tensor(mu, u)
    if device is not None:
        mu, u = mu.to(device), u.to(device)

    # Split data
    data_split = split_data(snapshots = u, split = split)
    
    return data_split, mesh



def save_analysis(analysis_dict : Optional[dict], filename : str):
    """ To save analysis files.

    Args:
        analysis_dict (Optional[dict]): the container for the analysis results.
        filename (str): the path to save.
    """
    with open(filename, 'wb') as outfile:
        pickle.dump(analysis_dict, outfile)



def load_analysis(filename : str):
    """ To load analysis files.

    Args:
        filename (str): the path to load.

    Returns:    
        the loaded container.
    """
    file = open(filename, 'rb')
    analysis_dict = pickle.load(file, encoding = "bytes")
    return analysis_dict



def is_not_decreasing(x : Sequence[int]) -> bool:
    """ Checks if the sequence is not decreasing

    Args:
        x (Sequence[int]): the sequence.
    
    Returns:
        the truth value.
    """
    return not np.prod(np.diff(np.flip(x)) >= 0)



def generate_data_for_tests_suite(device : str, ns : int = 800):
    """ Easy to generate data. Must be called after setting the seed for 
        reproducible behavior.

    Args:
        device (str): the device name to load the data to.
        ns (int): the total number of samples.
    
    Returns:
        the sampled data.
    """
    nh = 1001
    ntrain, nval = int(ns / 4), int(ns / 8)
    ntest = ns - (ntrain + nval)
    X = np.linspace(0, 1, nh)
    MU = 0.4 + 0.2 * np.random.rand(ns, 1)
    U = torch.tensor(
        np.array([np.exp(-100 * (X - mu)**2) for mu in MU]).astype('float32')
    ).to(device)
    utrain, uval, utest = U[:ntrain], U[ntrain:(ntrain+nval)], U[-ntest:]

    return utrain, uval, utest