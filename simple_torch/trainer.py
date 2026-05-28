import torch.distributed as dist
import torch
from torch.cuda.amp import autocast
from simple_torch.utils import concat_across_epoch, count_parameters, print_fn

class Trainer:
    def __init__(self, model, datamodule, logger, max_epochs):
        
        self.logger = logger
        self.model = model
        self.datamodule = datamodule
        self.max_epochs = max_epochs

        # select data loader
        print_fn("* Setting up SimpleTorch data module")
        print_fn(f"** Dataset: {datamodule.hparams.dataset}")
        datamodule.sanity_check()
        self.train_dataloader = datamodule.train_dataloader()
        self.val_dataloader = datamodule.eval_dataloader("val")
        self.test_dataloader = datamodule.eval_dataloader("test")

        # initialize model
        print_fn("* Setting up SimpleTorch module")
        print_fn(f"** Model: {model.hparams.model}")
        model.sanity_check()
        print_fn("** Trainable model parameters")
        count_parameters(model)
        model.build_model(logger)

        
    def fit(self):
        print_fn("="*20, " Training starts ", "="*20)
        for epoch in range(self.max_epochs):
            # train for one epoch
            self.logger.set_epoch(epoch)
            if dist.is_torchelastic_launched() is True:
                self.train_dataloader.sampler.set_epoch(epoch)

            self.training_one_epoch(epoch)
            self.eval_one_epoch(epoch, True)
            self.eval_one_epoch(epoch, False)

            # if epoch == 30:
            #     # save model at epoch 30
            #     torch.save(self.model.model.state_dict(), "model_pasa.pth")
                

        self.logger.end()
        # clean up ddp if possible
        if dist.is_torchelastic_launched() is True:
            dist.destroy_process_group()

    def training_one_epoch(self, epoch):
        # switch to train mode
        self.model.train()

        for batch_idx, data in enumerate(self.train_dataloader):
            print_fn(f" Epoch: {epoch} | Batch: {batch_idx}/{len(self.train_dataloader)-1}...")
            self.logger.new_step()

            if type(data) is not list:
                data = [data]
            # allocate all tensors on GPU
            data_on_device = [i.cuda() for i in data]

            with autocast():
                loss = self.model.training_step(*data_on_device)
            
            self.model.optimization_step(loss)


    def eval_one_epoch(self, epoch, is_val = True):

        if is_val == True:
            dataloader = self.val_dataloader
            eval_step = self.model.val_step
            eval_epoch_end = self.model.val_epoch_end
            print_fn(f"Evaluating epoch {epoch} on val set...")
        else:
            dataloader = self.test_dataloader
            eval_step = self.model.test_step
            eval_epoch_end = self.model.test_epoch_end
            print_fn(f"Evaluating epoch {epoch} on test set...")

        self.model.eval()

        outputs_ls = []

        with torch.no_grad():
            for _, data in enumerate(dataloader):

                if type(data) is not list:
                    data = [data]
                # allocate all tensors on GPU
                data_on_device = [i.cuda() for i in data]

                outputs = eval_step(*data_on_device)

                outputs_ls.append(outputs)
                
        outputs_concat = concat_across_epoch(outputs_ls, len(dataloader.dataset))

        eval_epoch_end(*outputs_concat)
        # syncronize all processes
        if dist.is_torchelastic_launched() is True:
            dist.barrier()


