import torch, os
import torchvision.transforms as tr 

from typing import Optional, List
from torch.optim.adam import Adam
from pytorch_lightning import LightningModule, Trainer
from torchvision.datasets import MNIST


from mypt.data.dataloaders.standard_dataloaders import initialize_train_dataloader
from mypt.linearBlocks.fully_connected_blocks import ExponentialFCBlock 
from mypt.code_utilities import pytorch_utilities as pu
from mypt.shortcuts import P
from mypt.data.datasets.genericFolderDs import GenericDsWrapper

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

class FcWrapper(LightningModule):
    def __init__(self, 
                 lr: float,
                 in_features:int,
                 num_layers: int, 
                 dropout: Optional [float],
                 ):
        super().__init__()
        
        self.lr = lr
        self.loss = torch.nn.CrossEntropyLoss() 
        self.model = ExponentialFCBlock(output=10, 
                                        in_features=in_features, 
                                        num_layers=num_layers,
                                        dropout=dropout
                                        )
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward(x)

    def configure_optimizers(self):
        return Adam(params=self.model.parameters(), lr=self.lr)

    def training_step(self, batch, *args, **kwargs):
        x, y = batch
        model_output = self.forward(x)
        return self.loss.forward(model_output, y)


# create a custom dataset around the MNIST dataset to see if multi-threading still works properly
class MnsitGenericWrapper(GenericDsWrapper):
    def __init__(self, 
                root_dir: P, 
                augmentations: List,
                train,
                samples_per_cls: Optional[int] = None) -> None:

        super().__init__(
                root_dir=root_dir,
                augmentations=augmentations,
                train=train)

        self._ds = MNIST(root=root_dir,     
                         train=True,
                         transform=tr.Compose(self.augmentations),
                         download=True)

        # call the self._set_samples_per_cls method after setting the self._ds field
        if samples_per_cls is not None:
            self.samples_per_cls_map = self._set_samples_per_cls(samples_per_cls)
            self._len = 10 * samples_per_cls
        else:
            self._len = len(self._ds)

    def __getitem__(self, index):
        return self._ds.__getitem__(index)

def run(data_folder: P):
    # _ds = MNIST(root=data_folder, 
    #       train=True, 
    #       transform=tr.Compose([tr.ToTensor(), tr.Lambda(lambda x : x.reshape(-1,))]), 
    #       download=True
    #       )
    
    # let's try to make it work with a custom dataset
    
    ds = MnsitGenericWrapper(root_dir=data_folder, 
                             train=True, 
                             samples_per_cls=None,
                             augmentations=[tr.ToTensor(), tr.Lambda(lambda x : x.reshape(-1,))]
                             )

    dl = initialize_train_dataloader(dataset_object=ds, 
                                     seed=0, 
                                     batch_size=512, 
                                     num_workers=0, 
                                     warning=False)  

    log_dir = os.path.join(SCRIPT_DIR, 'temp_logs')

    device = pu.get_default_device()

    wrapper = FcWrapper(lr=0.1, 
                        in_features=28 * 28, 
                        num_layers=3, 
                        dropout=0.1)

    trainer = Trainer(
                    accelerator='gpu' if 'cuda' in device else 'cpu',  
                    devices=2, 
                    # strategy='ddp',
                    logger=False, # no logging for now
                    default_root_dir=log_dir,
                    max_epochs=5,
                    check_val_every_n_epoch=2,
                    log_every_n_steps=3,
                    )

    trainer.fit(wrapper, train_dataloaders=dl)

if __name__ == '__main__':
    mnist_data = os.path.join(SCRIPT_DIR, 'data', 'mnist')
    run(mnist_data)
