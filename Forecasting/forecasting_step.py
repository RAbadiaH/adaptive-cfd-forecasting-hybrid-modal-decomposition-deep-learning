import matplotlib.pyplot as plt
import os

import torch

import lightning as L

import Forecasting.training_inference_module as training_inference_module
from Forecasting.nn_lstm import ModelLSTM
from Forecasting.model_trainer import load_weights

L.seed_everything(32)

#-------------------------- Auxiliary functions --------------------------#

def findPretrainedPath(folder_path, train_size, num_modes, version_num):

    try:
    
        dirList = os.listdir(folder_path)

    except:

        return 'None'

    for i1 in dirList:
        
        path = f'snaps_{train_size}_modes_{num_modes}'

        if path == i1:

            newPath = os.path.join(
                folder_path, 
                i1, 
                f'lstm/lightning_logs/version_{version_num}/checkpoints/'
            )

            try:

                newPath = os.path.join(
                    newPath, 
                    os.listdir(newPath)[0]
                )

                return newPath

            except:

                return 'None'
        
        else:

            continue

    return 'None'


def forecasting_step(data_ten, train_schedule, last_train_schedule_te, 
    num_preds, folder_path, version_num, inp_seq, out_seq, stride, step, 
    batch_size, epochs, model_name, device, optimizer_name, 
    optimizer_hparams):
    print("#####################################################")
    print(data_ten.shape[-1])
    print("#####################################################")
    #--------------------- Create tensor for predictions ---------------------#

    predictions = torch.empty([
        1,
        *data_ten.shape[:-1],
        num_preds
    ])

    #------------------------------ Perform SVD ------------------------------#

    print("\nPerforming SVD ...", end='\r')

    flattenShape    = torch.prod(torch.tensor(data_ten.shape[:-1]))

    U_list  = []
    S_list  = []
    VT_list = []

    mean_flow_list = torch.zeros([1, flattenShape, 1])

    # SVD for dataset
    dataset = torch.clone(data_ten)

    dataset = torch.reshape(
        dataset, 
        [flattenShape, dataset.shape[-1]]
    )

    mean_flow = torch.mean(dataset, dim = -1, keepdim = True)

    dataset = dataset - mean_flow

    U, S, VT = torch.linalg.svd(dataset, full_matrices = False)

    S = torch.diag(S)

    mean_flow_list[0, :, :] = mean_flow
    # U_list.append(U)
    # S_list.append(torch.diag(S))
    # VT_list.append(VT)

    print("Performing SVD: DONE\n")

    del dataset

    #-------------------------------- Training --------------------------------#

    _, num_modes = train_schedule['te'][0]
    num_snap = train_schedule['snap'][0]

    pretrained_path = findPretrainedPath(
        folder_path, 
        num_snap, 
        num_modes, 
        version_num
    )

    if (len(last_train_schedule_te.keys()) > 0):

        _, num_modes_last = last_train_schedule_te['te'][0] #i1
        last_num_snap = last_train_schedule_te['snap'][0]

        last_pretrained_path = findPretrainedPath(
            folder_path, 
            last_num_snap, 
            num_modes_last, 
            version_num
        )

    else:

        last_pretrained_path = None

    CHECKPOINT_PATH = folder_path + f"/snaps_{num_snap}_modes_{num_modes}/"

    # U   = torch.clone(U_list[-1][:, :num_modes])
    # S   = torch.clone(S_list[-1][:num_modes, :num_modes])
    # VT  = torch.clone(VT_list[-1][:num_modes, :])

    U = U[:, :num_modes]
    S = S[:num_modes, :num_modes]
    VT = VT[:num_modes, :]

    min_val     = VT.mean(dim = 1, keepdim = True)
    range_val   = VT.std(dim = 1, keepdim = True)

    train_set   = (torch.clone(VT) - min_val) / (range_val + torch.tensor(1e-7))

    model_hparams = {
        'num_modes':    num_modes,
        'act_fn':       torch.nn.ReLU(),
        'hidden_size':  100,
        'p':            out_seq
    }

    fitted_model = training_inference_module.training_module(
        train_set, 
        inp_seq, 
        out_seq, 
        stride, 
        step, 
        batch_size, 
        epochs, 
        model_name, 
        device, 
        CHECKPOINT_PATH, 
        pretrained_path, 
        last_pretrained_path, 
        num_modes, 
        model_hparams, 
        optimizer_name, 
        optimizer_hparams
    )

    #------------------------------ Forecasting ------------------------------#

    # No warm-up
    last_entry = train_set[:, -inp_seq -3:]

    ten_pred = training_inference_module.autoreg_pred(
        fitted_model, 
        last_entry, 
        num_preds, 
        out_seq,
        warmup_steps = 3
    )

    ten_pred = ten_pred.detach()
    ten_pred = ten_pred * range_val + min_val
    ten_pred = U @ S @ ten_pred

    ten_pred += mean_flow

    ten_pred = torch.reshape(
        ten_pred, 
        [*data_ten.shape[:-1], num_preds]
    )

    return ten_pred