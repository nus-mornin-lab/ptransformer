from typing import Any
import torch.distributed as dist
import os
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from simple_torch.utils import SequentialDistributedSampler, check_distributed_on, count_parameters, print_fn


class SimpleTorchDataModule():
    def __init__(self, hparams):
        
        self.hparams = hparams
        # initialize the empty datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.num_workers = 32
    
    def sanity_check(self):
        if self.train_dataset is None:
            raise Exception("No available traing dataset!")
        if self.val_dataset is None:
            raise Exception("No available val dataset!")
        if self.test_dataset is None:
            raise Exception("No available test dataset!")
        
        print_fn(f"Train dataset: {len(self.train_dataset)}")
        print_fn(f"Val dataset: {len(self.val_dataset)}")
        print_fn(f"Test dataset: {len(self.test_dataset)}")

    def train_dataloader(self):

        if dist.is_torchelastic_launched() is True:
            train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)
            train_dataloader = DataLoader(dataset=self.train_dataset, batch_size = self.hparams.batch_size,
                                       sampler=train_sampler, num_workers=self.num_workers, pin_memory=True)
        else:
            train_dataloader = DataLoader(self.train_dataset, self.hparams.batch_size, 
                                      shuffle=True, num_workers=self.num_workers, pin_memory=True)

        return train_dataloader

    def eval_dataloader(self, dataset_name):
        
        eval_dataset = self.val_dataset if dataset_name == "val" else self.test_dataset

        if dist.is_torchelastic_launched() is True:
            eval_sampler = SequentialDistributedSampler(eval_dataset, batch_size=self.hparams.batch_size)
            eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=self.hparams.batch_size,
                                      sampler=eval_sampler, num_workers=self.num_workers, pin_memory=True)
        else:
            eval_dataloader = DataLoader(eval_dataset, self.hparams.batch_size, num_workers=self.num_workers, pin_memory=True)

        return eval_dataloader
       
class SimpleTorchModule(nn.Module):
    def __init__(self, hparams) -> None:
        super().__init__()
        self.hparams = hparams
        self.logger = None

        self.model = None
        self.criterion = None
        self.optimizer = None

        self.metrics = None

        self.scaler = GradScaler()

    def sanity_check(self):
        if self.model is None:
            raise Exception("No available model!")
        if self.criterion is None:
            raise Exception("No available criterion!")
        if self.optimizer is None:
            raise Exception("No available optimizer!")

    def build_model(self, logger):
        self.logger = logger

        self.model.cuda()
        if self.metrics is not None:
            self.metrics.cuda()

        if dist.is_torchelastic_launched() is True:
            device = int(os.environ['LOCAL_RANK'])
            self.model = DDP(self.model, device_ids=[device], output_device=device)

    def save_model(self, save_path):
        if dist.is_torchelastic_launched() is True:
            if int(os.environ['LOCAL_RANK']) == 0:
                torch.save(self.model.module.state_dict(), save_path)
                print_fn(f"Model checkpoint saved on {save_path}!")
            dist.barrier()
        else:
            torch.save(self.model.state_dict(), save_path)

    
    def optimization_step(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def forward(self, *args: Any, **kwds: Any) -> Any:
        return self.model(*args, **kwds)

    def training_step(self, *args: Any, **kwds: Any):
        raise NotImplementedError()

    def val_step(self, *args: Any, **kwds: Any):
        raise NotImplementedError()
    
    def test_step(self, *args: Any, **kwds: Any):
        return self.val_step(*args, **kwds)
    
    def val_epoch_end(self, *args: Any, **kwds: Any):
        raise NotImplementedError()
    
    def test_epoch_end(self, *args: Any, **kwds: Any):
        raise NotImplementedError()

    def log(self, metrics):
        self.logger.log_metrics(metrics)


    
