"""Training loop and bookkeeping (directories, saving, versioned log folders). V. Balmer, 2026."""

import glob
import os
import pickle
import random
import re
import shutil

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from .architectures import FFNN, Adam_LBFGS, CustomLosses, build_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42

TRAINING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # training/
REPO_DIR = os.path.dirname(TRAINING_DIR)

_wandb = None          # wandb module, only imported (and used) when logging is switched on


def _log(d: dict):
    """wandb.log if logging is switched on (see main_train), else no-op."""
    if _wandb is not None:
        _wandb.log(d)


######################################## main function ########################################

def main_train(data: dict, inp: dict, save_path: str, logs_dir: str, save_folder: bool = True,
               stats: dict = None, log_wandb: bool = False, project_name: str = 'ShellSim3D'):
    """
    Train one model and (optionally) copy the results to a new versioned folder logs_dir/v_<n>.

    Args:
        data            (dict):     normalised torch data (X_train_tt, y_train_tt, X_eval_tt, y_eval_tt)
        inp             (dict):     hyperparameters (config_inp.inp)
        save_path       (str):      folder for the intermediate results (best / last model, inp, stats, test data)
        logs_dir        (str):      base folder of the versioned model folders
        save_folder     (bool):     if True, copy the results to logs_dir/v_<latest+1>
        stats           (dict):     normalisation statistics of this dataset
        log_wandb       (bool):     log the training to Weights & Biases
        project_name    (str):      wandb project name

    Returns:
        version         (int|None): number of the created version folder (None if save_folder is False)
    """
    global _wandb
    if log_wandb:
        import wandb
        _wandb = wandb
        wandb.init(config=inp, project=project_name)

    try:
        # 1 - Create model instance
        model = FFNN(inp)
        print(model)

        # 2 - Train and evaluate, save the best evaluated model
        model = simple_train(inp, model, data, save_path)
        torch.save(model.state_dict(), os.path.join(save_path, 'last_trained_model.pt'))
        model.eval()

        # 3 - Copy model and data to a new version folder
        version = None
        if save_folder:
            filenames = ['inp.pkl', 'stats.pkl', 'last_trained_model.pt', 'best_trained_model.pt', 'test_data.pkl']
            os.makedirs(logs_dir, exist_ok=True)
            version = copy_files_with_incremented_version(save_path, logs_dir, filenames)
    finally:
        if _wandb is not None:
            _wandb.finish()
            _wandb = None

    return version


def setup_dirs() -> tuple:
    """Resolve DATA_DIR / MODEL_DIR / LOGS_DIR from environment variables or the defaults
    (sampling/data, training/checkpoints, training/logs). MODEL_DIR holds the working files of the current run
    (inp, stats, test data, best / last model), which are copied to LOGS_DIR/v_<n> at the end."""
    path_data = os.environ.get('DATA_DIR') or os.path.join(REPO_DIR, 'sampling', 'data')
    MODEL_DIR = os.environ.get('MODEL_DIR') or os.path.join(TRAINING_DIR, 'checkpoints')
    LOGS_DIR  = os.environ.get('LOGS_DIR')  or os.path.join(TRAINING_DIR, 'logs')
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR,  exist_ok=True)
    print(f'[train] DATA_DIR  = {path_data}')
    print(f'[train] MODEL_DIR = {MODEL_DIR}')
    print(f'[train] LOGS_DIR  = {LOGS_DIR}')
    return path_data, MODEL_DIR, LOGS_DIR


def setup_hyperparams(inp: dict, sobolev: bool) -> dict:
    """Copy of the config with inp['Sobolev'] set and the env-var overrides NUM_EPOCHS, BATCH_SIZE,
    HIDDEN_LAYERS, EARLY_STOP_STEPS, EARLY_STOP_RTOL applied."""
    inp = dict(inp)
    inp['Sobolev'] = sobolev
    if os.environ.get('NUM_EPOCHS'):
        inp['num_epochs']    = int(os.environ['NUM_EPOCHS'])
        print(f'[train] NUM_EPOCHS override    -> {inp["num_epochs"]}')
    if os.environ.get('BATCH_SIZE'):
        inp['batch_size']    = int(os.environ['BATCH_SIZE'])
        print(f'[train] BATCH_SIZE override    -> {inp["batch_size"]}')
    if os.environ.get('HIDDEN_LAYERS'):
        inp['hidden_layers'] = os.environ['HIDDEN_LAYERS']
        print(f'[train] HIDDEN_LAYERS override -> {inp["hidden_layers"]}')
    if os.environ.get('EARLY_STOP_STEPS'):
        inp['early_stop_steps'] = int(os.environ['EARLY_STOP_STEPS'])
        print(f'[train] EARLY_STOP_STEPS override -> {inp["early_stop_steps"]}')
    if os.environ.get('EARLY_STOP_RTOL'):
        inp['early_stop_rtol'] = float(os.environ['EARLY_STOP_RTOL'])
        print(f'[train] EARLY_STOP_RTOL override  -> {inp["early_stop_rtol"]}')
    return inp


######################################## training ########################################

def simple_train(inp: dict, model: FFNN, data: dict, save_path: str) -> FFNN:
    """
    Set up optimiser, scheduler, loss and data loaders, then run the training loop.

    Args:
        inp         (dict):     hyperparameters
        model       (FFNN):     model instance
        data        (dict):     normalised torch data (X_train_tt, y_train_tt, X_eval_tt, y_eval_tt)
        save_path   (str):      folder where the best model is saved

    Returns:
        model       (FFNN):     trained model (state after the last epoch)
    """

    set_torch_params()

    optimizer = optimizer_setup(inp, data, model)
    scheduler = scheduler_setup(inp, optimizer)
    lossFn = build_loss(inp['loss_type'], inp)

    model.to(device)

    if inp['batch_size'] is None:
        batch_size_train = data['X_train_tt'].shape[0]
        batch_size_eval  = data['X_eval_tt'].shape[0]
    else:
        batch_size_train = inp['batch_size']
        batch_size_eval  = inp['batch_size']
    train_loader = DataLoader(TensorDataset(data['X_train_tt'], data['y_train_tt']), batch_size=batch_size_train, shuffle=True)
    eval_loader  = DataLoader(TensorDataset(data['X_eval_tt'], data['y_eval_tt']), batch_size=batch_size_eval, shuffle=True)

    return simple_train_aux_(train_loader, eval_loader, model, optimizer, scheduler, lossFn,
                             inp['num_epochs'], save_path, inp)


def simple_train_aux_(train_loader, eval_loader, model, optimizer, scheduler, lossFn, MAXEPOCHS, save_path, inp):
    """
    Training loop. Optional early stopping on the epoch-mean validation loss: stop once it has not improved by
    more than early_stop_rtol (relative) for early_stop_steps optimiser steps (early_stop_steps None -> off).
    """
    best_val_loss = float('inf')
    es_steps = inp.get('early_stop_steps')
    es_rtol  = inp.get('early_stop_rtol', 1e-3)
    es_best, es_since = float('inf'), 0
    for epoch in range(MAXEPOCHS):
        model.train()

        batch = 0
        epoch_loss = 0
        for X_train_batch, y_train_batch in train_loader:
            X_torch, y_torch = X_train_batch.to(device), y_train_batch.to(device)

            def closure(backward=True):
                optimizer.zero_grad()

                if inp['Sobolev']:
                    custom_crit = CustomLosses(inp, model)
                    loss, loss_logs = custom_crit.Sobolev_CustomLoss(X_torch, y_torch)
                else:
                    pred = model(X_torch)
                    loss = lossFn(pred, y_torch[:, 0:inp['out_size']])
                    loss_logs = {'train_loss': loss}

                if backward:
                    loss.backward()

                return loss, loss_logs

            optimizer.zero_grad()
            loss, loss_logs = optimizer.step(closure)

            epoch_loss += loss.item()

            if inp['lr_scheduler'] == 'standard':
                scheduler.step()

            _log(loss_logs)
            _log({'Learning Rate': scheduler.get_last_lr()[0]})
            _log({'epoch': epoch})
            batch = batch+1

            if epoch % 100 == 0 or batch % int(len(train_loader)/2) == 0:
                print(f"Epoch [{epoch}/{MAXEPOCHS}] and Batch [{batch}/{len(train_loader)}] with LR {scheduler.get_last_lr()[0]:.2e}: \tLoss: {loss.item():.4e}")
            if batch == int(len(train_loader))-1:
                _log({'Average train loss per epoch': epoch_loss / batch})

        model, best_val_loss, mean_val_loss = simple_eval(inp, model, eval_loader, train_loader, lossFn, scheduler,
                                                          epoch, MAXEPOCHS, best_val_loss, save_path)

        if es_steps is not None:
            if mean_val_loss < es_best * (1 - es_rtol):
                es_best, es_since = mean_val_loss, 0
            else:
                es_since += batch
            if es_since >= es_steps:
                print(f'[early stop] epoch {epoch}: mean val loss {mean_val_loss:.4e} not improved by '
                      f'>{es_rtol:g} (rel.) for {es_since} steps (best {es_best:.4e}).')
                break

    return model


def simple_eval(inp, model, eval_loader, train_loader, lossFn, scheduler, epoch, MAXEPOCHS, best_val_loss, save_path):
    """Validation pass of one epoch; saves the model whenever the batch validation loss improves on the best one."""

    epoch_val_loss = 0
    val_batch = 0

    for X_eval_batch, y_eval_batch in eval_loader:
        X_eval, y_eval = X_eval_batch.to(device), y_eval_batch.to(device)
        model.eval()
        with torch.no_grad():
            if not inp['Sobolev']:
                pred_val = model(X_eval)
                val_loss = lossFn(pred_val, y_eval[:, 0:inp['out_size']])
            else:
                custom_crit = CustomLosses(inp, model)
                val_loss, _ = custom_crit.Sobolev_CustomLoss(X_eval, y_eval)

            _log({"val_loss": val_loss})
            epoch_val_loss += val_loss.item()
            val_batch += 1

            if epoch % 100 == 0 or val_batch % max(1, len(eval_loader) // 2) == 0:
                print(f"Epoch [{epoch}/{MAXEPOCHS}] and Batch [{val_batch}/{len(eval_loader)}]: \tValidation Loss: {val_loss.item():.4e}")

        if val_batch == int(len(train_loader))-1:
            _log({'Average evaluation loss per epoch': epoch_val_loss / val_batch})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _log({"best_val_loss": best_val_loss})
            save_best_model(model, epoch, val_loss, save_dir=save_path)

        if inp['lr_scheduler'] == 'plateau':
            scheduler.step(val_loss)

    return model, best_val_loss, epoch_val_loss / max(val_batch, 1)


def optimizer_setup(inp: dict, data: dict, model: FFNN) -> Adam_LBFGS:
    """Adam_LBFGS; switch to LBFGS after switch_step_percentage of all optimiser steps (1 -> never switch)."""
    if inp['switch_step_percentage'] == 1:
        no_switch_step = float('inf')
    elif inp['batch_size'] is None:
        no_switch_step = int(inp['switch_step_percentage']*inp['num_epochs'])
    else:
        n_train = data['X_train_tt'].shape[0]
        no_switch_step = int(inp['switch_step_percentage']*inp['num_epochs']*int(n_train/inp['batch_size']))
    return Adam_LBFGS(inp, model.parameters(), no_switch_step)


def scheduler_setup(inp: dict, optimizer: Adam_LBFGS):
    """'standard': StepLR (x0.99 every 100 steps); 'plateau': ReduceLROnPlateau on the validation loss."""
    if inp['lr_scheduler'] == 'standard':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer.adam, step_size=100, gamma=0.99)
    elif inp['lr_scheduler'] == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer.adam, 'min')
    return scheduler


def save_best_model(model, epoch, val_loss, save_dir):
    """Save best_trained_model__<epoch>.pt and remove the best models of earlier epochs."""
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, f'best_trained_model__{epoch}.pt'))
    if epoch > 200:
        print(f'Best model saved at epoch {epoch} with validation loss: {val_loss:.4f}')

    model_pattern = re.compile(r"best_trained_model__(\d+)\.pt")
    for filename in os.listdir(save_dir):
        match = model_pattern.match(filename)
        if match and int(match.group(1)) < epoch:
            os.remove(os.path.join(save_dir, filename))
    return


######################################## auxiliary functions ########################################

def copy_files_with_incremented_version(src_folder: str, base_dest_folder: str, files_to_copy: list) -> int:
    """
    Copy files_to_copy from src_folder to base_dest_folder/v_<latest+1>. 'best_trained_model.pt' stands for the
    best_trained_model__<epoch>.pt file, which is moved instead of copied.

    Returns:
        version (int): number of the new version folder
    """
    version = get_latest_version_folder(base_dest_folder) + 1
    new_folder_path = os.path.join(base_dest_folder, f"v_{version}")
    os.makedirs(new_folder_path, exist_ok=True)

    for file_name in files_to_copy:
        if file_name == 'best_trained_model.pt':
            model_files = glob.glob(os.path.join(src_folder, 'best_trained_model_*.pt'))
            if not model_files:
                print(f"{file_name} does not exist in the source folder.")
            for src_path in model_files:
                shutil.copy2(src_path, os.path.join(new_folder_path, os.path.basename(src_path)))
                os.remove(src_path)
        else:
            src_path = os.path.join(src_folder, file_name)
            if os.path.exists(src_path):
                shutil.copy2(src_path, os.path.join(new_folder_path, file_name))
            else:
                print(f"{file_name} does not exist in the source folder.")

    print(f"Files copied to {new_folder_path}")
    return version


def get_latest_version_folder(base_folder):
    """Highest n of the folders 'v_<n>' in base_folder (0 if there are none)."""
    version_numbers = [int(m.group(1)) for f in os.listdir(base_folder) if (m := re.match(r'v_(\d+)$', f))]
    return max(version_numbers) if version_numbers else 0


def save_inp(inp: dict, save_path: str):
    """Save the hyperparameters (inp.pkl) for testing / inference."""
    with open(os.path.join(save_path, 'inp.pkl'), 'wb') as fp:
        pickle.dump(inp, fp)


def save_stats(stats: dict, save_path: str):
    """Save the normalisation statistics (stats.pkl) for testing / inference."""
    with open(os.path.join(save_path, 'stats.pkl'), 'wb') as fp:
        pickle.dump(stats, fp)


def save_test_data(train_eval_test_data: dict, save_path: str):
    """Save the test split (test_data.pkl: X_test, y_test, physical units)."""
    test_data = {'X_test': train_eval_test_data['X_test'],
                 'y_test': train_eval_test_data['y_test']}
    with open(os.path.join(save_path, 'test_data.pkl'), 'wb') as fp:
        pickle.dump(test_data, fp)


def set_torch_params():
    """Seed python, numpy and torch and make cuDNN deterministic."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
