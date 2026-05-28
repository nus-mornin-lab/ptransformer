import os
import comet_ml
from simple_torch.trainer import Trainer
from simple_torch.utils import prepare_training
from utils.logger import create_logger
from data.data_entry import create_data_module
from models.model_entry import create_st_module
from options import get_all_hparams



def main():
    # Import hyperparameters
    general_conf_path = "./configs/general.yaml"

    hparams = get_all_hparams(general_conf_path)

    prepare_training(hparams)

    # Initialize logger, data module, model and trainer
    logger = create_logger(hparams)

    data_module = create_data_module(hparams)

    model = create_st_module(hparams)

    trainer = Trainer(
        model = model, 
        datamodule = data_module, 
        logger = logger, 
        max_epochs = hparams.max_epochs
    )
    
    trainer.fit()

if __name__ == "__main__":
    main()


