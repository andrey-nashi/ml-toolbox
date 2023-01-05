from pytorch_lightning import LightningModule

class VortexModel(LightningModule):

    def __init__(self, model):
        pass

    def forward(self, x):
        return x

    def training_set(self, batch, batch_idx):
        return

