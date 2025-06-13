import torch
import torch.nn as nn



class BiorthogonalPairs(nn.Module):
    """ Class to encode and decode with a pair of biorthogonal matrices.
    """

    def __init__(self, n : int, r : int):
        """

        Args:
            n (int): the number of dofs.
            r (int): the low-rank dimension (must be < n).
        """
        super().__init__()
        self.orth_param = nn.utils.parametrizations.orthogonal
        assert r < n
        d = min(r, n - r)
        self.X = nn.Linear(n, r + d, bias = False)
        self.U = nn.Linear(r, r, bias = False)
        self.eig = nn.Parameter(torch.ones(r, ))
        self.Z = nn.Linear(r, r, bias = False)
        self.Q = nn.Parameter(torch.zeros(d, r))
        self.r = r


    def finalize_init(self):
        """ To be called after initializing the matrices to enforce orthogonal
            parameterization before training.
        """
        self.orth_param(self.X)
        self.orth_param(self.U)
        self.orth_param(self.Z)
    

    def enc(self, x):
        """ Encode function.

        Args:
            x: the input.
        
        Returns:
            the encoded input.
        """
        S = self.eig**2
        enc_mat = self.X.weight.T[:,:self.r] @ \
            (torch.einsum('ij,j->ij', self.U.weight.T, S) @ self.Z.weight)
        return x @ enc_mat


    def dec(self, x):
        """ Decode function.

        Args:
            x: the input.
        
        Returns:
            the decoded input.
        """
        S = self.eig**2
        dec_mat = self.X.weight.T[:,:self.r] @ \
            (torch.einsum('ij,j->ij', self.U.weight.T, S**(-1)) @ \
             self.Z.weight) + self.X.weight.T[:,self.r:] @ self.Q
        return x @ dec_mat.T



class Normalizer:
    """ Normalizer to facilitate training convergence.
    """

    def __init__(self, utrain : torch.tensor):
        """
        
        Args:
            utrain (torch.tensor): the training snapshots.
        
        """
        self.min_, self.max_ = torch.min(utrain), torch.max(utrain)
        

    def forward(self, u_):
        """ Forward pass.

        Args:
            u_: the (unnormalized) input.
        
        Returns: 
            the normalized input.
        """
        return (u_ - self.min_) / (self.max_ - self.min_)
    

    def backward(self, u_):
        """ Backward pass.

        Args:
            u_: the (normalized) output.
        
        Returns: 
            the (de-normalized) output.
        """
        return u_ * (self.max_ - self.min_) + self.min_
    