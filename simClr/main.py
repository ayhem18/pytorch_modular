import os

from model_train.models.resnet.model import ResnetSimClr
from model_train.train import train
from mypt.code_utilities import pytorch_utilities as pu

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

if __name__ == '__main__':
    train_data_folder = os.path.join(SCRIPT_DIR, 'data', 'plant_seedlings', 'unlabeled_data')

    model = ResnetSimClr(input_shape=(3, 224, 224), output_dim= 256, num_fc_layers=3, dropout=0.3)

    train(model=model, 
          train_data_folder=train_data_folder, 
          val_data_folder=None,
          num_epochs=150, 
          batch_size=128, 
          temperature=0.5, 
          seed=0)

