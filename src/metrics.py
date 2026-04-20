from math import sqrt
import torch



def squared_error(
    utrue : torch.tensor, 
    upred : torch.tensor
):
    """ Batch-wise squared error w.r.t the euclidean metric"""
    return (utrue - upred).pow(2).sum(axis = -1)


def relative_error(
    utrue : torch.tensor, 
    upred : torch.tensor,
):
    """ Batch-wise relative error w.r.t the euclidean metric"""
    num = (utrue - upred).pow(2).sum(axis = 1).sqrt() 
    den = (utrue).pow(2).sum(axis = 1).sqrt()
    return num / den


def mse(
    utrue : torch.tensor, 
    upred : torch.tensor
):
    """ Mean Squared Error"""
    return squared_error(utrue = utrue, upred = upred).mean()


def mre(
    utrue : torch.tensor, 
    upred : torch.tensor
):
    """ Mean Relative Error"""
    return relative_error(utrue = utrue, upred = upred).mean()


def band95_squared_error(
    utrue : torch.tensor, 
    upred : torch.tensor
):
    """ Band width for the 95% confidence interval of the squared error."""
    ns = utrue.shape[0]
    return 1.96 * squared_error(utrue = utrue, upred = upred).std() / sqrt(ns)


def band95_relative_error(
    utrue : torch.tensor, 
    upred : torch.tensor
):
    """ Band width for the 95% confidence interval of the relative error."""
    ns = utrue.shape[0]
    return 1.96 * relative_error(utrue = utrue, upred = upred).std() / sqrt(ns)

