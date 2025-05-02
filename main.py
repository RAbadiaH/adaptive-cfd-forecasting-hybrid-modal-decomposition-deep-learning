import matplotlib.pyplot as plt
import os
import shutil
import time

import torch

from load_data import load_data

from Forecasting.forecasting_step  import forecasting_step

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#---------------------------- Set hyperparameters ----------------------------#
""" Training Hyperparameters"""
case_name           = 'laminar' # 'JetLES' or 'laminar' or 'turbulent'
num_modes           = 5 # 13 turbulent, 5 jet, 6 laminar
epochs              = 1500
batch_size          = 8

""" Rollling Window Hyperparameters"""
inp_seq             = 10
out_seq             = 1
stride              = 0
step                = 1

""" DL Model Hyperparameters"""
model_name          = 'lstm' # DO NOT MODIFY
optimizer_name      = 'Adam' # DO NOT MODIFY
optimizer_hparams   = {"lr": 1e-3, "weight_decay": 1e-4}

""" Adaptive Framework Hyperparameters"""
initial_snap        = 100 # S_0
new_data_solver     = 100  # S_1
num_preds           = 200 # P

PATH = f"./{case_name}_saved_models_{initial_snap}_{num_preds}_{new_data_solver}"
VERSION = 0

time_1 = time.time()

#--------------------------------- Load data ---------------------------------#

data_orig = load_data(case_name)
data_orig = torch.from_numpy(data_orig).to(device).type(torch.float32)

data_ten = torch.clone(data_orig[..., :initial_snap])
solution = torch.clone(data_ten)
print(data_ten.shape)

new_idx = data_ten.shape[-1]

#---------------------------------- Adaptive ----------------------------------#

last_train_schedule_te = {}

count = 0

while (data_ten.shape[-1] < data_orig.shape[-1]):

    num_snap = data_ten.shape[-1]

    train_schedule = {
        'te':   [[num_snap, num_modes]],
        'snap': [solution.shape[-1]]
    }

    data_ten = forecasting_step(
        data_ten, 
        train_schedule,  
        last_train_schedule_te, 
        num_preds, 
        PATH, 
        VERSION, 
        inp_seq, 
        out_seq, 
        stride, 
        step, 
        batch_size, 
        epochs, 
        model_name, 
        device, 
        optimizer_name, 
        optimizer_hparams
    )

    if (data_ten != None):
    
        solution = torch.cat(
            [solution, data_ten.clone()],
            dim = -1
        )

    last_idx = solution.shape[-1]

    print(f"\nLAST IDX: {last_idx}\n")

    if ((last_idx - new_idx) >= 3):

        last_train_schedule_te = train_schedule.copy()

        count += 1

        # num_preds = 500

    data_ten = data_orig[..., last_idx : last_idx + new_data_solver].clone()

    solution = torch.cat(
        [solution, data_ten.clone()],
        dim = -1
    )

    new_idx = solution.shape[-1]

    print(f"\nNEW IDX: {new_idx}\n")

    if (data_ten.shape[-1] < new_data_solver):

        break

torch.save(solution, f'{case_name}_predicted_ten.pt')

time_2 = time.time()

print(f"Time elapsed - {time_2 - time_1} seconds.")
