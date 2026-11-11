# Hyperparameters for training (train.py). The defaults are those of the model in trained_models/v_156.

inp = {
    # network
    'input_size': 13,                               # 6 strains + 7 geom [t, rho_xtop, rho_xbot, rho_ytop, rho_ybot, fc, fs]
    'out_size': 6,                                  # generalised stresses [n_x, n_y, n_xy, m_x, m_y, m_xy]
    'hidden_layers': str([1024]*5),
    'activation': 'ELU',                            # any torch.nn activation class name
    'dropout_rate': 0,
    'BatchNorm': False,

    # optimisation
    'batch_size': 4096,                             # None: full-batch training
    'num_epochs': 80,
    'learning_rate': 0.001,
    'lr_scheduler': 'standard',                     # 'standard': StepLR (x0.99 every 100 steps); 'plateau': ReduceLROnPlateau
    'switch_step_percentage': 1,                    # fraction of the steps after which Adam switches to LBFGS (1: never)
    'early_stop_steps': None,                       # stop if the epoch-mean val loss has not improved for this many steps (None: off)
    'early_stop_rtol': 1e-3,                        # relative improvement that counts for early stopping

    # loss
    'loss_type': 'wbMSELoss',                       # 'MSELoss' or 'wbMSELoss' (up-weights near-zero moments and Dmb/Dbm entries)
    'loss_type_D': None,                            # loss for the stiffness term; None: same as loss_type
    'weight': 20.0,                                 # wbMSELoss: up-weight factor
    'band': 0.5,                                    # wbMSELoss: |normalised target| below which entries are up-weighted
    'w_s': str([0.9, 0.1]),                         # Sobolev weights [w1 (stresses), w2 (stiffness)], or 'max'

    # normalisation
    't_normalise': True,                            # divide the outputs by t^power before the standardisation (needs t as input)
    't_norm_powers': [1, 1, 1, 2, 2, 2],            # output t-power per component [n_x, n_y, n_xy, m_x, m_y, m_xy]; D row j uses the same power
    't_idx': 6,                                     # column of t in the input vector
    't_norm_input_powers': [0, 0, 0, 1, 1, 1],      # input t-power per strain (curvatures ~ 1/t); D column k adds this power
    'fc_normalise': True,                           # also divide all stresses and stiffnesses by the sample's fc
    'fc_idx': 11,                                   # column of fc in the input vector
}
