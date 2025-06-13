
from dlroms import *
import numpy as np
import pickle
from typing import *



def loadexp(meshpath : str, datapath : str, device = None):
    """ Loads mesh and data given their paths.

    Args:
        meshpath (str): path/to/mesh.
        datapath (str): path/to/data.
    
    Returns:
        split data (train, val, test), and the mesh.
    """
    mesh = fe.loadmesh(meshpath)
    data = np.load(datapath)
    mu, u = data['mu'], data['u']
    ndata, _  = mu.shape
    ndata, _ = u.shape
    mu, u = dv.tensor(mu, u)
    ntrain = ndata//2
    nval = ndata//4
    utrain = u[:ntrain]
    uval = u[ntrain:ntrain+nval]
    utest = u[ntrain+nval:]
    data_split = (utrain, uval, utest)
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





