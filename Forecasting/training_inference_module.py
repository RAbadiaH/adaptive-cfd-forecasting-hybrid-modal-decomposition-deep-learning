import Forecasting.preprocessing as preprocessing
import Forecasting.model_trainer as model_trainer

import torch
import torch.utils.data as data

from lightning import seed_everything

def training_module(train_set, inp_seq, out_seq, stride, 
    step, batch_size, epochs, model_name, device, 
    CHECKPOINT_PATH, pretrained_path, last_pretrained_path, num_modes, 
    model_hparams, optimizer_name, optimizer_hparams):

    seed_everything(42)

    # Create datasets using the temporal modes.
    dataset_tr = preprocessing.RollingWinDataset(
        train_set, 
        inp_seq = inp_seq, 
        out_seq = out_seq, 
        stride  = stride, 
        step    = step
    )

    # Create dataloaders using the datasets.
    dl_tr = data.DataLoader(
        dataset_tr, 
        batch_size  = batch_size, 
        shuffle     = True, 
        drop_last   = True,
        num_workers = 0
    )
    
    dl_tr_entire = data.DataLoader(
        dataset_tr, 
        batch_size  = len(dataset_tr), 
        shuffle     = False, 
        drop_last   = False,
        num_workers = 0
    )

    # Start training procedure.
    fitted_model, saved_weights_path = model_trainer.train_model(
        epochs                  = epochs, 
        model_name              = model_name, 
        train_loader            = dl_tr, 
        device                  = device, 
        CHECKPOINT_PATH         = CHECKPOINT_PATH, 
        pretrained_path         = pretrained_path, 
        last_pretrained_path    = last_pretrained_path, 
        num_modes               = num_modes, 
        model_hparams           = model_hparams, 
        optimizer_name          = optimizer_name, 
        optimizer_hparams       = optimizer_hparams
    )

    return fitted_model

def autoreg_pred(model, vt_mat, num_preds, out_seq, warmup_steps):

    preds_mat = torch.zeros([vt_mat.shape[0], num_preds * out_seq])

    # in_model = torch.clone(vt_mat[:, :-warmup_steps])
    # in_model = in_model.T[None, ...]

    # # Warm up steps
    # preds, (hn, cn) = model.predict_step(in_model)

    # for i1 in range(1, warmup_steps):

    #     in_model = torch.clone(vt_mat[:, i1:-warmup_steps + i1])
    #     in_model = in_model.T[None, ...]

    #     preds, (hn, cn) = model.predict_step(in_model, hn, cn)

    in_model = torch.clone(vt_mat[:, warmup_steps:])
    in_model = in_model.T[None, ...]

    # Predict new data
    for i1 in range(num_preds):

        preds, (hn, cn) = model.predict_step(in_model)

        preds_ = torch.clone(preds[0, :, :])
        preds_mat[:, out_seq * i1 : out_seq * (i1+1)] = preds_.T

        in_model = torch.cat([in_model[:, out_seq:, :], preds], dim = 1)

    return preds_mat