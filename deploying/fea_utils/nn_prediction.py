"""NN predictions for the FEA: loads a trained model (training/utils/predictor.NNPredictor) and evaluates the
generalised stresses and / or tangents at the Gauss points. V. Balmer, 2026."""

import numpy as np

from training.utils.predictor import NNPredictor


def load_NN_model(model_dir: str, version) -> NNPredictor:
    """
    Load a trained model once (best_trained_model_*.pt, inp.pkl, stats.pkl of model_dir/v_<version>).

    Args:
        model_dir   (str):      folder with the model versions, e.g. training/trained_models
        version     (int|str):  model version

    Returns:
        NNPredictor: model with its hyperparameters and normalisation statistics
    """
    return NNPredictor.from_folder(model_dir, version)


def n_geom_from_model(model: NNPredictor) -> int:
    """
    Number of geometry / material input columns after the 6 strains (inp['input_size'] - 6).

    Args:
        model   (NNPredictor):  trained model

    Returns:
        int: 0 (strains only), 1 (rho_y), 2 (t, rho_y), 3 (t, rho_x, rho_y), 5 (t, rho_xtop, rho_xbot, rho_ytop,
             rho_ybot), 6 (... + fc) or 7 (... + fc, fs)
    """
    return model.inp['input_size'] - 6


def make_NN_prediction(input_j: np.ndarray, predict: str, model: NNPredictor, non_batchwise: bool = False,
                       symm_predict: bool = False, strain_jac_only: bool = False) -> dict:
    """
    Stresses and / or tangents of the NN in physical units [N, mm].

    Args:
        input_j         (np.arr):       inputs [6 generalised strains, geometry], shape (n, input_size)
        predict         (str):          'sig', 'D' or 'sig_D' (both from one CPU pass)
        model           (NNPredictor):  trained model
        non_batchwise   (bool):         tangent in one pass instead of batchwise (inp['batch_size'])
        symm_predict    (bool):         mid-plane symmetry wrapper g(x) = 0.5 [f(x) + R_out f(R_in x)]
        strain_jac_only (bool):         unused (the tangent is always taken w.r.t. the 6 strains only)

    Returns:
        dict: 'sig' (n, 6) and 'D' (n, 6, 6); None for the part that is not predicted
    """
    symmetry = 'midplane' if symm_predict else None
    if predict == 'sig':
        return model.predict(input_j, D=False, symmetry=symmetry)
    if predict == 'D':
        return model.predict(input_j, sig=False, symmetry=symmetry, batched=not non_batchwise)
    if predict == 'sig_D':
        return model.predict(input_j, symmetry=symmetry, one_pass=True)
    raise ValueError(f"predict must be 'sig', 'D' or 'sig_D', got {predict!r}")
