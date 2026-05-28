from datetime import datetime
import json
import shutil
import os
import sys
# from torch.utils.tensorboard import SummaryWriter
from comet_ml import Experiment
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast
import torch.distributed as dist
from simple_torch.utils import print_fn

class CometNewExperiment(Experiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.curr_step = -1

    def new_step(self):
        self.curr_step += 1

def create_logger(hparams):

    login_info = {"api_key": "xx", 
                  "project_name": "xxx", 
                  "workspace": "xxx"}

    if (dist.is_torchelastic_launched() is True and int(os.environ['LOCAL_RANK']) == 0) or (dist.is_torchelastic_launched() is not True):
        login_info["disabled"] = not hparams.logger
        if hparams.logger:
            print_fn("\t* Initializing logger...")
    else:
        login_info["disabled"] = True

    experiment = CometNewExperiment(**login_info)
    experiment.log_parameters(hparams)
    experiment.log_code(folder="./")
    
    return experiment
   