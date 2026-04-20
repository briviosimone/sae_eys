from abc import ABC
from collections.abc import Iterable

import numpy as np
import torch



class BilipActivation(ABC):
    """ Abstract base class for bilipschitz activations.
    """

    @property
    def sharpness(self):
        """ Computes and store the sharpness value
        """
        return self.lip_act * self.lip_invact - 1

   
    def setup(self):
        self.gain_act = np.sqrt(
            2 / (self.lip_invact**(-2) + self.lip_act**2)
        ) 
        self.gain_invact = np.sqrt(
            2 / (self.lip_act**(-2) + self.lip_invact**2)
        ) 



class LeakyReLU(BilipActivation, ABC):
    """ Implementation of (alpha,beta)-Leaky-ReLU.
    """
    
    def __init__(self, alpha : float, beta : float = 1.):
        """

        Args:
            alpha (float): the negative slope.
            beta (float): the positive slope (defaults to 1).
        """
        
        if (alpha < 0) or (beta < 0):
            raise ValueError('Both alpha and beta should be > 0.')
        self.alpha = alpha
        self.beta = beta
        super().__init__()
        self.act = lambda x: self.alpha * x * (x <= 0) + \
            self.beta * x * (x > 0)
        self.invact = lambda x: self.alpha**(-1) * x * (x <= 0) +  \
            (self.beta)**(-1) * x * (x > 0)
        self.lip_act = max(self.beta, self.alpha)
        self.lip_invact = max((1.0 / self.beta), (1.0 / self.alpha)) 




class HypAct(BilipActivation, ABC):
    """ Implementation of the theta-Hyperbolic activation function.
        See: Otto et al. "Learning nonlinear projections for reduced-order
        modeling of dynamical systems using constrained
        autoencoders". Chaos (2023).
    """

    def __init__(self, alpha):
        """

        Args:
            alpha (float): the hyperbole parameter.
        """
        if (alpha < 0) or (alpha > np.pi / 4):
            raise ValueError('Allowed values for alpha are 0 < alpha < pi / 4.')
        self.alpha = alpha
        super().__init__()
        csc = lambda x: 1 / np.sin(x)
        sec = lambda x: 1 / np.cos(x)
        a = csc(self.alpha)**2 - sec(self.alpha)**2
        b = csc(self.alpha)**2 + sec(self.alpha)**2
        t1 = lambda x: b * x / a 
        t2 = lambda x: np.sqrt(2) / a * csc(self.alpha)
        t3 = lambda x: 2 * x * csc(self.alpha) * sec(self.alpha)
        t4 = lambda x: np.sqrt(2) * sec(self.alpha)
        self.act = lambda x: \
            t1(x) - t2(x) + 1 / a * ((t3(x) - t4(x))**2 + 2 * a)**(1/2)
        self.invact = lambda x: \
            t1(x) + t2(x) - 1 / a * ((t3(x) + t4(x))**2 + 2 * a)**(1/2)
        self.lip_act = (1 + np.tan(self.alpha)) / (1 - np.tan(self.alpha))
        self.lip_invact = self.lip_act




class BilipActivationConfig:
    """ Utility class to be used for analysis.
    """

    def __init__(
        self,
        name : str,
        bilipactivation : BilipActivation,
        parameters : Iterable[float],
        sharpnesses : Iterable[float]
    ):
        """

        Args:
            name (str): the activation name.
            bilipactivation (BilipActivation): the bilipschitz nonlinearity.
            parameters (Iterable[float]): the activation parameters to use in
                                          the analysis.
            sharpnesses (Iterable[float]): the activation sharpnesses to use in
                                           the analysis.
        """
        self.name = name
        self.bilipactivation = bilipactivation
        self.parameters = parameters
        self.sharpnesses = sharpnesses