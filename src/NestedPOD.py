
from .activations import BilipActivation
from scipy.linalg import svd
import numpy as np
import torch 


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

        # Extract hidden length
        self.hidden_len = len(red_dims) - 1

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

            for k in range(self.hidden_len):
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
        return torch.tensor(self.Ws[level][:proj_dim]), \
                torch.tensor(self.bs[level])
    

    def bounds(self, usnap : np.array):
        """ Obtain lower and upper bounds.

        Args:
            usnap (np.array): the snapshot matrix.

        Returns:
            lower bound, upper bound.
        """

        def Jaux(mat, proj, mean):
            proj_err = np.mean(
                np.sum(
                    (mat.T - (mat.T - mean.T) @ proj.T @ proj - mean.T)**2, 
                    axis = -1
                )
            )
            return proj_err

        latent_vector = usnap.T
        ub_val = 0.
        lb_val = 0.
        for k in range(len(self.Ws)):
            to_add = Jaux(latent_vector, self.Ws[k], self.bs[k].T)
            ub_val = ub_val + self.bilipactivation.lip_invact**(2 * k) * to_add
            lb_val = lb_val + self.bilipactivation.lip_act**(-2 * k) * to_add
            latent_vector = self.bilipactivation.act(
                self.Ws[k] @ (latent_vector - self.bs[k].T)
            )

        return lb_val, ub_val


        
