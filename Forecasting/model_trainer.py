import os

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

import torch
import torch.nn as nn
import torch.optim as optim

import Forecasting.nn_lstm as nn_lstm

#----------------------------- Models dictionary -----------------------------#

model_dict = {}

model_dict["lstm"] = nn_lstm.ModelLSTM

def create_model(model_name, model_hparams):

    if model_name in model_dict:

        return model_dict[model_name](**model_hparams)

    else:
        
        assert False, f"Unknown model name \"{model_name}\". Available models are: {str(model_dict.keys())}"

#------------------------------ LightningModule ------------------------------#

class HybridModule(L.LightningModule):

    def __init__(self, model_name, model_hparams, num_modes, optimizer_name, 
        optimizer_hparams):

        super().__init__()

        self.save_hyperparameters()

        self.model = create_model(model_name, self.hparams.model_hparams)

        self.loss_module = nn.MSELoss()

        self.example_input_array = torch.zeros((1, 10, num_modes), dtype = torch.float32)

    def forward(self, inp):
        
        return self.model(inp)

    def configure_optimizers(self):
        
        if self.hparams.optimizer_name == "Adam":
        
            optimizer = optim.AdamW(
                self.parameters(), 
                **self.hparams.optimizer_hparams
            )

        elif self.hparams.optimizer_name == "SGD":

            optimizer = optim.SGD(
                self.parameters(), 
                **self.hparams.optimizer_hparams
            )

        else:

            assert False, f"Unknown optimizer: \"{self.optimizer_name}\""

        # Warm-up of gradients.
        def lr_lambda(epoch):
            if epoch < 5:
            
                return 1.0  # Keep lr at 1e-5
            
            elif (epoch >= 5 and epoch < 100):

                return 1e-3 / 1e-5 # Scaling factor to increase lr to 1e-3
            
            elif (epoch >= 100 and epoch < 150):

                return 1e-4 / 1e-5

            else:
                return 1.0

        # scheduler = optim.lr_scheduler.LambdaLR(
        #     optimizer,
        #     lr_lambda = lr_lambda
        # )

        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0 = 100, #100
            T_mult = 2, #1
            eta_min = 1e-6
        )

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):

        inputs, targets = batch
        
        preds, _ = self.model(inputs)
        
        loss = self.loss_module(preds, targets)
        mae = nn.L1Loss()(preds, targets)
        
        self.log('train_mae', mae, on_step = False, on_epoch = True, prog_bar = True)
        self.log('train_loss', loss, on_step = False, on_epoch = True, prog_bar = True)
        
        return loss

    def predict_step(self, inputs, h0 = None, c0 = None):
        
        preds, (h_n, c_n) = self.model(inputs, h0, c0)

        return preds, (h_n, c_n)

#----------------------------------- TRAINER -----------------------------------

def train_model(epochs, model_name, train_loader, device, 
    CHECKPOINT_PATH, pretrained_path, last_pretrained_path, **kwargs):

    trainer = L.Trainer(
        default_root_dir    = os.path.join(CHECKPOINT_PATH, model_name),
        accelerator         = "gpu" if str(device).startswith("cuda") else "cpu",
        devices             = 1,
        max_epochs          = epochs,
        log_every_n_steps   = 10,
        callbacks           = [
            ModelCheckpoint(
                save_weights_only   = True, 
                mode                = "min", 
                monitor             = "train_loss"
            ), 
            LearningRateMonitor("epoch")
        ],
        enable_progress_bar = True)

    trainer.logger._log_graph           = True
    trainer.logger._default_hp_metric   = None

    # Check whether pretrained model exists. If yes, load it and skip training    
    if os.path.isfile(pretrained_path):

        print("\n###################################")
        print(f"Found pretrained model, loading...")
        print("###################################\n")

        # Automatically loads the model with the saved hyperparameters
        modeL = HybridModule.load_from_checkpoint(pretrained_path)

        return modeL, pretrained_path

    else:

        if (last_pretrained_path == None):

            modeL = HybridModule(model_name = model_name, **kwargs)

        else:

            modeL = HybridModule.load_from_checkpoint(last_pretrained_path)

        trainer.fit(modeL, train_loader)

        modeL = HybridModule.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )

        saved_weights_path = trainer.checkpoint_callback.best_model_path

        return modeL, saved_weights_path
    
def load_weights(model, pretrained_path):

    modeL = HybridModule.load_from_checkpoint(pretrained_path, model = model)

    return modeL