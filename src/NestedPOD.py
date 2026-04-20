from scipy.linalg import svd
import torch 

from activations import BilipActivation
from utils import is_not_decreasing


class NestedPOD:

    def __init__(
        self,
        snapshots, 
        red_dims : list, 
        bilipactivation : BilipActivation,
    ):
        """ Creates the Nested POD AE architecture.

        Args:
            snapshots: the training snapshot matrix.
            red_dims (list): a list comprising the number of neurons in each 
                             level.
            bilipactivation (BilipActivation): a bilipschitz activation.
        
        Returns:
            the Nested POD AE architecture and its weights matrices
        """

        # Stores the bilipactivation
        self.bilipactivation = bilipactivation 
        self.red_dims = red_dims
        self.n_samples = snapshots.shape[0]

        # Checks
        if is_not_decreasing(red_dims):
            raise ValueError('red_dims = {red_dims} is not decreasing')


        # Extract hidden length
        self.num_levels = len(red_dims) - 1

        # Aux functions to compute POD
        def compute_pod_matrix(mat):
            return svd(mat.cpu().numpy().T, full_matrices = False)[0].T
        
        # Aux function to compute POD at each level
        def compute_internal_pods(snapshots, act):

            latent_vector = snapshots.cpu()

            def center_snapshots(vec_):
                mean_ = latent_vector.mean(axis = 0).reshape(1,-1)
                return vec_ - mean_, mean_
            
            Vs_pod_ae = list()
            bs_pod_ae = list()

            for k in range(self.num_levels):
                latent_vector, latent_mean = center_snapshots(latent_vector)
                bs_pod_ae.append(latent_mean.cpu().numpy())
                Vs_pod_ae.append(compute_pod_matrix(latent_vector))
                proj_mat = Vs_pod_ae[-1][:red_dims[k+1]]
                latent_vector = act(latent_vector @ proj_mat.T)
            return Vs_pod_ae, bs_pod_ae

        # Get weight matrices through nested POD approach
        self.Ws, self.bs = compute_internal_pods(
            snapshots, 
            bilipactivation.act
        )


    def get_affine_transform(self, level : int, proj_dim : int):
        """ Obtain level-wise weights and biases

        Args:
            level (int): the encoding level
            proj_dim (int): the reduced dimension 

        Returns:
            weight matrix, bias vector (as torch.tensor)
        """
        num_basis = self.Ws[level].shape[0]
        if proj_dim > self.Ws[level].shape[0]:
            raise ValueError(f'Proj. dim. = {proj_dim} cannot exceed, ' \
            f'num. basis functions = {num_basis}')
        return torch.tensor(self.Ws[level][:proj_dim]), \
                torch.tensor(self.bs[level])
    


        
