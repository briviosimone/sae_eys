from dlroms import *
import numpy as np
import torch
import torch.nn as nn
from .utils import *
from .blocks import *
from .activations import *
from .NestedPOD import NestedPOD
from scipy.stats import ortho_group
from collections.abc import *



class AE(torch.nn.Module):
    """ The classical autoencoder, with mirrored architecture, using the 
        'decoder activation' in every layer except the last.
    """

    def __init__(
        self,
        red_dims : Iterable[int], 
        bilipactivation : BilipActivation
    ):
        """ 

        Args:
            red_dims (Iterable[int]): list comprising the number of neurons for 
                                      each level, from the outermost (the number
                                      of dofs) to the innermost (the latent
                                      space dimension).
            bilipactivation (BilipActivation): the bilipschitz activation. 
        """
        super().__init__()
        self.n_levels = len(red_dims) - 1
        self.weights_enc = [
            nn.Linear(red_dims[j-1], red_dims[j], bias = True)
            for j in range(1,len(red_dims))
        ]
        self.weights_enc = nn.ParameterList(self.weights_enc)
        self.weights_dec = [
            nn.Linear(red_dims[-j], red_dims[-j-1], bias = True)
            for j in range(1,len(red_dims))
        ]
        self.weights_dec = nn.ParameterList(self.weights_dec)
        self.bilipactivation = bilipactivation
        self.red_dims = red_dims
        

    def standard(self):
        """ Standard initialization procedure (He normal).
        """
        for k in range(self.n_levels):
            gain = np.sqrt(2 / self.bilipactivation.he_slope_inv)
            shape_dec, shape_enc = self.weights_enc[k].weight.shape
            std_enc = gain / np.sqrt(shape_enc)
            std_dec = gain / np.sqrt(shape_dec)
            nn.init.normal_(self.weights_enc[k].weight, 0.0, std_enc)
            nn.init.normal_(self.weights_dec[k].weight, 0.0, std_dec)
            if self.weights_enc[k].bias is not None:
                nn.init.constant_(self.weights_enc[k].bias, 0)
            if self.weights_dec[k].bias is not None:
                nn.init.constant_(self.weights_dec[k].bias, 0)
        

    def eys(self, nested_pod : NestedPOD):
        """ EYS initialization procedure.

        Args:
            nested_pod (NestedPOD): used to initialize the weight matrices.
        """
        for k in range(self.n_levels):
            W, b = nested_pod.get_affine_transform(
                level = k, 
                proj_dim = self.red_dims[k+1]
            )
            self.weights_enc[k].weight.data = W
            self.weights_dec[-k-1].weight.data = W.T
            self.weights_enc[k].bias.data = (-b[0]) @ W.T
            self.weights_dec[-k-1].bias.data = b[0]


    def forward(self, x):
        """ Forward pass.

        Args:
            x: the input.
        
        Returns:
            the AE output.
        """
        for level in range(self.n_levels):
            x = self.bilipactivation.invact(self.weights_enc[level](x))
        for level in range(self.n_levels - 1):
            x = self.bilipactivation.invact(self.weights_dec[level](x))
        return self.weights_dec[self.n_levels - 1](x)



class SAE(AE):
    """ The symmetric autoencoder, with mirrored architecture, using the 
        a bilipschitz activation in the encoder and its inverse in the decoder.
    """

    def __init__(
        self,
        red_dims : Iterable[int], 
        bilipactivation : BilipActivation
    ):
        """ 

        Args:
            red_dims (Iterable[int]): list comprising the number of neurons for 
                                      each level, from the outermost (the number
                                      of dofs) to the innermost (the latent
                                      space dimension).
            bilipactivation (BilipActivation): the bilipschitz activation. 
        """
        super().__init__(red_dims, bilipactivation)
       

    def standard(self):
        """ Standard initialization procedure (He normal).
        """
        for k in range(self.n_levels):
            gain_enc = np.sqrt(2 / self.bilipactivation.he_slope)
            gain_dec = np.sqrt(2 / self.bilipactivation.he_slope_inv)
            shape_dec, shape_enc = self.weights_enc[k].weight.shape
            std_enc = gain_enc / np.sqrt(shape_enc)
            std_dec = gain_dec / np.sqrt(shape_dec)
            nn.init.normal_(self.weights_enc[k].weight, 0.0, std_enc)
            nn.init.normal_(self.weights_dec[k].weight, 0.0, std_dec)
            if self.weights_enc[k].bias is not None:
                nn.init.constant_(self.weights_enc[k].bias, 0)
            if self.weights_dec[k].bias is not None:
                nn.init.constant_(self.weights_dec[k].bias, 0)


    def forward(self, x):
        """ Forward pass.

        Args:
            x: the input.
        
        Returns:
            the AE output.
        """
        for level in range(self.n_levels - 1):
            x = self.bilipactivation.act(self.weights_enc[level](x))
        x = self.weights_enc[self.n_levels - 1](x)
        for level in range(self.n_levels - 1):
            x = self.bilipactivation.invact(self.weights_dec[level](x))
        return self.weights_dec[self.n_levels - 1](x)



class SBAE(torch.nn.Module):
    """ The symmetric orthogonal autoencoder, with mirrored architecture, using 
        a bilipschitz activation in the encoder and its inverse in the decoder, 
        with the encoder weights biorthogonal to the decoder weights and 
        opposite biases.
    """

    def __init__(
        self,
        red_dims : Iterable[int], 
        bilipactivation : BilipActivation
    ):
        """ 

        Args:
            red_dims (Iterable[int]): list comprising the number of neurons for 
                                      each level, from the outermost (the number
                                      of dofs) to the innermost (the latent
                                      space dimension).
            bilipactivation (BilipActivation): the bilipschitz activation. 
        """
        super().__init__()
        self.n_levels = len(red_dims) - 1
        self.weights = [
            BiorthogonalPairs(red_dims[j-1], red_dims[j]) 
            for j in range(1,len(red_dims))
        ]
        self.weights = nn.ParameterList(self.weights)
        self.biases = [
            nn.parameter.Parameter(torch.zeros(red_dims[j]))
            for j in range(len(red_dims) - 1)
        ]
        self.biases = nn.ParameterList(self.biases)
        self.activation = bilipactivation
        self.red_dims = red_dims
    

    def standard(self):
        """ Standard initialization procedure (see Otto et al., 2023).
        """
        for k in range(self.n_levels):
            dim1, dim2 = self.weights[k].X.weight.shape
            square_mat = ortho_group.rvs(dim = dim2).astype(np.float32)
            self.weights[k].X.weight.data = torch.tensor(square_mat[:dim1])
            self.biases[k].data = torch.zeros_like(
                self.biases[k].data
            )
            r = self.weights[k].U.weight.shape[0]
            self.weights[k].U.weight.data = torch.eye(r)
            self.weights[k].Z.weight.data = torch.eye(r)
            self.weights[k].finalize_init()


    def eys(self, nested_pod : NestedPOD):
        """ EYS initialization procedure adapted to the overparametrization
            induced by the Biorthogonal pairs implementation.

        Args:
            nested_pod (NestedPOD): used to initialize the weight matrices.
        """
        for k in range(self.n_levels):
            W, b = nested_pod.get_affine_transform(
                level = k, 
                proj_dim = min(2 * self.red_dims[k+1], self.red_dims[k])
            )
            #dim1, dim2 = self.weights[k].X.weight.data.shape
            self.weights[k].X.weight.data = W
            self.biases[k].data = b[0]
            r = self.weights[k].U.weight.shape[0]
            self.weights[k].U.weight.data = torch.eye(r)
            self.weights[k].Z.weight.data = torch.eye(r)
            self.weights[k].finalize_init()

    
    def forward(self, x):
        """ Forward pass.

        Args:
            x: the input.
        
        Returns:
            the AE output.
        """
        for level in range(self.n_levels - 1):
            x = self.activation.act(
                self.weights[level].enc(x - self.biases[level])
            )
        x = self.weights[self.n_levels - 1].enc(
            x - self.biases[self.n_levels - 1]
        )
        for level in range(self.n_levels - 1):
            x = self.activation.invact(
                self.weights[-level-1].dec(x) + self.biases[-level-1]
            )
        x = self.weights[0].dec(x) + self.biases[0]
        return x



class SOAE(torch.nn.Module):
    """ The symmetric orthogonal autoencoder, with mirrored architecture, using 
        a bilipschitz activation in the encoder and its inverse in the decoder, 
        with orthogonal encoder weights which are equal to the decoder weights,
        and opposite biases.
    """
    

    def __init__(
        self,
        red_dims : Iterable[int], 
        bilipactivation : BilipActivation
    ):
        """ 

        Args:
            red_dims (Iterable[int]): list comprising the number of neurons for 
                                      each level, from the outermost (the number
                                      of dofs) to the innermost (the latent
                                      space dimension).
            bilipactivation (BilipActivation): the bilipschitz activation. 
        """
        super().__init__()
        self.orth_param = torch.nn.utils.parametrizations.orthogonal
        self.n_levels = len(red_dims) - 1
        self.weights = [
            nn.Linear(red_dims[j-1], red_dims[j], bias = False)
            for j in range(1,len(red_dims))
        ]
        self.weights = nn.ParameterList(self.weights)
        self.biases = [
            nn.parameter.Parameter(torch.zeros(red_dims[j]))
            for j in range(len(red_dims) - 1)
        ]
        self.biases = nn.ParameterList(self.biases)
        self.activation = bilipactivation
        self.red_dims = red_dims


    def standard(self):
        """ Standard initialization procedure (see Otto et al., 2023).
        """
        for k in range(self.n_levels):
            dim1, dim2 = self.weights[k].weight.shape
            square_mat = ortho_group.rvs(dim = dim2).astype(np.float32)
            self.weights[k].weight.data = torch.tensor(square_mat[:dim1,:])
            self.weights[k] = self.orth_param(self.weights[k])
            self.biases[k].data = torch.zeros_like(
                self.biases[k].data
            )
            

    def eys(self, nested_pod : NestedPOD):
        """ EYS initialization procedure.

        Args:
            nested_pod (NestedPOD): used to initialize the weight matrices.
        """
        for k in range(self.n_levels):
            W, b = nested_pod.get_affine_transform(
                level = k, 
                proj_dim = self.red_dims[k+1]
            )
            self.weights[k].weight.data = W
            self.weights[k] = self.orth_param(self.weights[k])
            self.biases[k].data = b[0]


    def forward(self, x):
        """ Forward pass.

        Args:
            x: the input.
        
        Returns:
            the AE output.
        """
        for level in range(self.n_levels - 1):
            x = self.activation.act(self.weights[level](x - self.biases[level]))
        x = self.weights[self.n_levels - 1](x - self.biases[self.n_levels - 1])
        for level in range(self.n_levels - 1):
            x = self.activation.invact(
                x @ self.weights[-level-1].weight + self.biases[-level-1]
            )
        x = x @ self.weights[0].weight + self.biases[0]
        return x
    



