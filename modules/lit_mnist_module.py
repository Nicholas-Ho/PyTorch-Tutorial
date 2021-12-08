from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule

from torch.optim import Adam

from argparse import ArgumentParser

# Pytorch Autoencoder
class LitMNISTModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        
        # Defining the dense layers (to be used in the model definition in forward(x))
        # MNIST Images are (1, 28, 28) (channels, height, width)
        self.layer_1 = nn.Linear(28 * 28, 128)
        self.layer_2 = nn.Linear(128, 256)
        self.layer_3 = nn.Linear(256, 10)
        
    # Defining the model, Tensorflow-style. Returns prediction
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)
        
        x = F.log_softmax(x, dim=1)
        return x
    
    # Training-loop logic
    def training_step(self, batch, batch_idx):
        x, y = batch # Load training data
        logits = self(x) # Predict y based on training data's x
        loss = F.nll_loss(logits, y) # Compute loss
        self.log("Loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    # Validation logic
    def validation_step(self, batch, batch_idx):
        x, y = batch # Load validation data
        logits = self(x) # Predict y based on training data's x
        loss = F.nll_loss(logits, y) # Compute loss
        self.log("Loss", loss, on_step=True, on_epoch=True)
        return loss
    
    # Testing logic
    def test_step(self, batch, batch_idx):
        x, y = batch # Load validation data
        logits = self(x) # Predict y based on training data's x
        loss = F.nll_loss(logits, y) # Compute loss
        self.log("Loss", loss, on_step=True, on_epoch=True, logger=True)
        return loss
    
    # Adam optimiser
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.args.learning_rate)

    # Argument Parsing for CLI
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitMNIST")
        parser.add_argument("--learning_rate", type=float, default=0.001)
        return parent_parser