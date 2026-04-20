import os
import tqdm
from typing import *

import torch 
import torch.nn as nn
import numpy as np

import metrics



def train(
    U : torch.utils.data.DataLoader, 
    model : nn.Module, 
    epochs : int, 
    optimizer : torch.optim.Optimizer, 
    U_val : torch.tensor, 
    patience : int = np.inf,
    scheduler = None,
    verbose : bool = True, 
    monitor_history : bool = False, 
    ckpt_path : str = 'best.pt'
):
    """ Function to train (with early stopping).

    Args:
        U (torch.utils.data.DataLoader): dataset of train snapshots.
        model (Optional[nn.Module]): the trainable model.
        epochs (int): the maximum number of epochs.
        U_val (torch.tensor): the validation data.
        patience (int): the early stopping patience (defaults to np.inf).
        scheduler: the learning rate scheduler.
        verbose (bool): if True, prints all the info in sequence, otherwise
                        uses tqdm to update console logging (defaults to True).
        monitor_history (bool): If True, the train function returns the training 
                                history (defaults to False).
        ckpt_path (str): used to save the model (defaults to 'best.pt').
    
    Returns:
        the trained model (and the training history, if monitor_history is 
        True).
    """

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
        loss = 0.0
        for ubatch in U:
            def closure():
                optimizer.zero_grad()
                upred = model(ubatch)
                loss = metrics.mse(utrue = ubatch, upred = upred)
                loss.backward()
                return loss

            loss += optimizer.step(closure)
        
        loss = loss / len(U)


        # Monitoring and validation (with early stopping)      
        with torch.no_grad():
            U_val_pred = model(U_val)
            loss_val = metrics.mse(utrue = U_val, upred = U_val_pred)
            if scheduler is not None:
                scheduler.step()
            if loss_val < loss_val_best:
                patience_counter = 0
                loss_val_best = loss_val
                torch.save(model.state_dict(), ckpt_path)
            else:
                patience_counter += 1

            # Updating history
            if monitor_history:
                history_monitor['loss_train'].append(loss.item())
                history_monitor['loss_val'].append(loss_val.item())

            # Logging
            if verbose:
                print("Epoch\t%d.\ntrain loss\nval loss\t%.2e" \
                      % (i+1, loss.item(), loss_val.item()))
            else:
                pbar.set_postfix(
                    {
                        'train loss': loss.item(), 
                        'val loss' : loss_val.item()
                    }
                )
            
        # Early stopping
        if patience_counter == patience:
            break

    # Loading best model
    model.load(ckpt_path = ckpt_path)
    model.eval()

    if monitor_history:
        return model, history_monitor
    else:
        return model
