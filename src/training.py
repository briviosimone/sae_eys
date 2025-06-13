import torch 
import torch.nn as nn
import numpy as np
import os
import tqdm
from typing import *



def classic_ae_mse(
    U : torch.tensor, 
    model: Optional[nn.Module]
):
    """ MSE for autoencoders.

    Args:
        U (torch.tensor): the snapshots.
        model (Optional[nn.Module]): the autoencoder.

    Returns:
        the MSE loss value.
    """
    U_hat = model(U)
    loss = (U - U_hat).pow(2).sum(axis = -1).mean()
    return loss



def train(
    J : callable, 
    U : Union[torch.tensor, torch.utils.data.DataLoader], 
    model : Optional[nn.Module], 
    epochs : int, 
    optimizer : torch.optim.Optimizer, 
    U_val : torch.tensor, 
    patience : int = np.inf,
    verbose : bool = True, 
    monitor_history : bool = False, 
    model_name : str = 'best'
):
    """ Function to train (with early stopping).

    Args:
        J (callable): the loss function.
        U (Union[torch.tensor, torch.utils.data.DataLoader]): train snapshots.
        model (Optional[nn.Module]): the trainable model.
        epochs (int): the maximum number of epochs.
        U_val (torch.tensor): the validation dataset.
        patience (int): the early stopping patience (defaults to np.inf).
        verbose (bool): if True, prints all the info in sequence, otherwise
                        uses tqdm to update console logging (defaults to True).
        monitor_history (bool): If True, the train function returns the training 
                                history (defaults to False).
        model_name (str): used to save the model (defaults to 'best').
    
    Returns:
        the trained model (and the training history, if monitor_history is 
        True).
    """

    # Setting savepath
    savepath = 'results'
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    savedir = os.path.join(savepath, model_name + '.pt')

    # Setting monitor metrics
    if monitor_history:
        history_monitor = dict()
        history_monitor['loss_train'] = list()
        history_monitor['loss_val'] = list()

    # Initialize quantities
    pbar = tqdm.tqdm(range(epochs))
    patience_counter = 0
    loss_val_best = np.inf

    # Training loop
    for i in pbar:

        # Optimization
        if isinstance(U, torch.utils.data.DataLoader):
            loss = 0.0
            for ubatch in U:
                def closure():
                    optimizer.zero_grad()
                    loss = J(ubatch, model)
                    loss.backward()
                    return loss

                loss += optimizer.step(closure)
            
            loss = loss / len(U)

        else:
            def closure():
                optimizer.zero_grad()
                loss = J(U, model)
                loss.backward()
                return loss

            loss = optimizer.step(closure)

        # Monitoring and validation (with early stopping)      
        with torch.no_grad():
            loss_val = J(U_val, model)
            if loss_val < loss_val_best:
                patience_counter = 0
                loss_val_best = loss_val
                if isinstance(model, nn.Module):
                    torch.save(model.state_dict(), savedir)
                else:
                    torch.save(model, savedir)
            else:
                patience_counter += 1

            if monitor_history:
                history_monitor['loss_train'].append(loss.item())
                history_monitor['loss_val'].append(loss_val.item())

            if verbose:
                print("Epoch\t%d.\nLoss\t%.2e" % (i+1, loss.item()))
            else:
                pbar.set_postfix(
                    {
                        'train loss': loss.item(), 
                        'val loss' : loss_val.item()
                    }
                )
            
        if patience_counter == patience:
            break

    # Saving
    if isinstance(model, nn.Module):
        model.load_state_dict(torch.load(savedir, weights_only = True))
        model.eval()
    else:
        model = torch.load(savedir)
    if monitor_history:
        return model, history_monitor
    else:
        return model
