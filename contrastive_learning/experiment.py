import os
import numpy as np
import torch
import torch.optim as optim
import pytorch_lightning as pl
from emb_models.base import BaseVAE

class GRNVAEExperiment(pl.LightningModule):
    def __init__(self, vae_model: BaseVAE, params: dict) -> None:
        super(GRNVAEExperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.save_hyperparameters(params)

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        gene_expr, adj_matrix = batch
        self.curr_device = gene_expr.device
        print("gene_expr.shape: ", gene_expr.shape)
        results = self.forward(gene_expr)
        train_loss = self.model.loss_function(*results,
                                              M_N=self.params['kld_weight'],
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)

        self.log_dict({f"train {key}": val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['total loss']

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        gene_expr, adj_matrix = batch
        self.curr_device = gene_expr.device

        results = self.forward(gene_expr)
        val_loss = self.model.loss_function(*results,
                                            M_N=1,
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)

        self.log_dict({f"valid {key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def on_validation_end(self) -> None:
        self.sample_genes()
    
    def on_fit_start(self):
        self.logger.log_hyperparams(self.hparams)

    def sample_genes(self):
        # Get sample reconstruction
        test_input, _ = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        recons = self.model.generate(test_input)
        np.save(os.path.join(self.logger.log_dir, 
                             "Reconstructions", 
                             f"recons_Epoch_{self.current_epoch}.npy"),
                recons.cpu().numpy())

        try:
            samples = self.model.sample(100, self.curr_device)
            np.save(os.path.join(self.logger.log_dir, 
                                 "Samples", 
                                 f"samples_Epoch_{self.current_epoch}.npy"),
                    samples.cpu().numpy())
        except Warning:
            pass

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])

        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                     gamma=self.params['scheduler_gamma'])

        return [optimizer], [scheduler]