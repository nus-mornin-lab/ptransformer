from models.trans_medprompts.base_cls import TransMedPromptsClsSTM
from utils.torch_utils import param_reset




def create_st_module(hparams, *args, **kwargs):
    # trans_medprompts
    if hparams.model in [ "trans_medprompts_mimic_mortality"]: 
        model = TransMedPromptsClsSTM(hparams)
    else:
        raise Exception("No model available.")

    return model