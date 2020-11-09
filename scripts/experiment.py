#!/usr/bin/env python
import argparse
import io
import numpy as np
import time
import torch
import tqdm
import ttpp

from copy import deepcopy


def get_name(args):
    name = f'{args.dataset}_{args.model_name}_{args.n_knots}'
    name += f'_{args.n_blocks}_{args.block_size}'
    name += f'_{args.hidden_size}'
    name += f'_{args.learning_rate}_{args.weight_decay}_{time.time()}'
    return name


def save(model):
    if issubclass(type(model), torch.jit.ScriptModule):
        return model.save_to_buffer()
    elif issubclass(type(model), torch.nn.Module):
        return deepcopy(model)
    else:
        raise RuntimeError(f'Cannot save type {type(model)}')


def load(cache):
    if issubclass(type(cache), torch.nn.Module):
        cache.init()
        return cache
    elif issubclass(type(cache), bytes):
        return torch.jit.load(io.BytesIO(cache))


def main(dataset,
         model_name,
         epochs=5000,
         learning_rate=1e-2,
         weight_decay=0,
         patience=300,
         max_grad_norm=2.,
         batch_size=2048,
         use_jit=False,
         return_model=False,
         **kwargs):
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    ## Load data
    dset = ttpp.data.load_dataset(dataset)
    d_train, d_val, d_test = dset.train_val_test_split(train_size=0.6, val_size=0.2, test_size=0.2)
    dl_train = torch.utils.data.DataLoader(d_train, batch_size=batch_size, shuffle=True)
    dl_val = torch.utils.data.DataLoader(d_val, batch_size=len(d_val), shuffle=False)
    dl_test = torch.utils.data.DataLoader(d_test, batch_size=len(d_test), shuffle=False)

    # Lets overwrite the data loader in case we do gradient descent without batches
    if batch_size >= len(dl_train):
        dl_train = [next(iter(dl_train))]
    
    # Load the whole validation set to reduce overhead at every iteration
    val_batch = next(iter(dl_val))

    ## Train model
    model = getattr(ttpp.models, model_name)(t_max=dset.t_max,
                                             lambda_init=d_train.mean_number_items,
                                             **kwargs)
    if use_jit:
        if model_name == 'Autoregressive':
            raise NotImplementedError("Jit is not support for RNN.")
        model = torch.jit.script(model)
    
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=100, verbose=True)

    # Tracking the best model
    impatient, threshold = 0, 1e-4
    best_loss, best_train = np.inf, np.inf
    # Storing model
    best_model = save(model)
    training_val_losses = []
    training_losses = []

    # Postfix for tqdm
    postfix = {
        'loss_train': 0,
        'loss_val': 0
    }

    training_start = time.time()
    with tqdm.tqdm(range(epochs), dynamic_ncols=True, postfix=postfix) as t:
        for epoch in t:
            # Optimization
            model.train()
            for batch in dl_train:
                opt.zero_grad()
                x, mask = batch[0], batch[1]
                loss = -(model.log_prob(x, mask) / d_train.mean_number_items).mean()
                loss.backward()
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()

            loss = loss.item()  # Convert to python scalar for the rest of the script
            training_losses.append(loss)
            lr_scheduler.step(loss)  # reduce the learning rate if our training loss stops decreasing

            # Validation
            model.eval()
            with torch.no_grad():
                x, mask = val_batch[0], val_batch[1]
                loss_val = -(model.log_prob(x, mask) / d_train.mean_number_items).mean().item()
                training_val_losses.append(loss_val)

            # Early stopping
            if (best_loss - loss_val) < threshold:
                impatient += 1
                if loss_val < best_loss:
                    best_train, best_loss = loss, loss_val
                    best_model = save(model)
            else:
                best_train, best_loss = loss, loss_val
                best_model = save(model)
                impatient = 0

            if impatient >= patience:
                print(f'Breaking due to early stopping at epoch {epoch}')
                break
            
            # Progressbar updates
            postfix['loss_train'] = loss
            postfix['loss_val'] = loss_val
            t.set_postfix(postfix)
    training_end = time.time()

    ## Test model
    model = load(best_model)
    model.eval()

    batch = next(iter(dl_test))
    x, mask = batch[0], batch[1]
    test_loss = -(model.log_prob(x, mask) / d_train.mean_number_items).mean().item()

    ## Construct results
    results = { 
        'test_loss': test_loss, 
        'val_loss': best_loss,
        'train_loss': best_train,
        'training_losses': training_losses,
        'training_val_losses': training_val_losses, 
        'final_epoch': epoch,
        'training_time': training_end - training_start
    }

    if return_model:
        return model, results
    else:
        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('model_name')
    # Training
    parser.add_argument('-b', '--batch_size', default=2048, type=int)
    parser.add_argument('-lr', '--learning_rate', default=0.01, type=float)
    parser.add_argument('-mg', '--max_grad_norm', default=2., type=float)
    parser.add_argument('-wd', '--weight_decay', default=1e-4, type=float)
    parser.add_argument('-e', '--epochs', default=5000, type=int)
    parser.add_argument('-p', '--patience', default=300, type=int)
    # Splines
    parser.add_argument('-nk', '--n_knots', default=20, type=int)
    parser.add_argument('-so', '--spline_order', default=2, type=int)
    # Block diagonal
    parser.add_argument('-bs', '--block_size', default=16, type=int)
    parser.add_argument('-nb', '--n_blocks', default=4, type=int)
    # RNN 
    parser.add_argument('-hs', '--hidden_size', default=32, type=int)
    # Logging
    parser.add_argument('-lp', '--log_params', action='store_true')
    parser.add_argument('-nc', '--enable_nan_check', action='store_true')
    parser.add_argument('--use_jit', action='store_true')
    parser.add_argument('-rm', '--return_model', action='store_true')

    args = parser.parse_args()

    result = main(**vars(args))
    print('Validation loss', result['val_loss'])
