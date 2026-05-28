import random
import numpy as np
import torch.distributed as dist
import os
import torch
import math
from prettytable import PrettyTable

def check_distributed_on(func):
    def wrapper(*args, **kwargs):
        if (dist.is_torchelastic_launched() is True and int(os.environ['LOCAL_RANK']) == 0) or (dist.is_torchelastic_launched() is not True):
            return func(*args, **kwargs)
    return wrapper

@check_distributed_on
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            params = parameter.numel()
        else:
            params = 0
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

@check_distributed_on
def print_fn(*args, **kwargs):
    """print in either normal mode or distributed mode
    """
    print(*args, **kwargs)

def concat_across_epoch(input_tensors_ls, num_total_examples):

    # concate one output
    if torch.is_tensor(input_tensors_ls[0]):
        return [concat_across_epoch_one_list(input_tensors_ls, num_total_examples)]
    # concate multiple outputs
    else:
        output_tensors_ls = []
        for input_tensors in zip(*input_tensors_ls):
            output_tensors_ls.append(concat_across_epoch_one_list(input_tensors, num_total_examples))

        return output_tensors_ls


def concat_across_epoch_one_list(batch_tensors_ls, num_total_examples):

    batch_tensors = torch.cat(batch_tensors_ls, dim=0)
    if dist.is_torchelastic_launched() is True:
        output_tensors_ls = [batch_tensors.clone() for _ in range(dist.get_world_size())]
        torch.distributed.all_gather(output_tensors_ls, batch_tensors)
        concat_tensors = torch.cat(output_tensors_ls, dim=0)
        # truncate the dummy elements added by SequentialDistributedSampler
        return concat_tensors[:num_total_examples]
    else:
        return batch_tensors


class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples

def prepare_training(hparams):

    print_fn("="*20, " Preparing for training ", "="*20)
    # Set random seed
    random.seed(hparams.seed)
    np.random.seed(hparams.seed)
    torch.manual_seed(hparams.seed)

    # check if the script is run in distributed mode
    if dist.is_torchelastic_launched() is True:
        print_fn("* Initializing distributed learning")
        os.environ["CUDA_VISIBLE_DEVICES"] = hparams.devices
        device = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl')
        print(f"Local rank {device} created!")
    else:
        device = int(hparams.devices.split(",")[0])

    # set the cuda device
    torch.cuda.set_device(device)