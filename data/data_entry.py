from data.trans_medprompts import TransMedPromptsSTDM



def create_data_module(hparams):

    # trans_medprompts
    if hparams.model in ["trans_medprompts_mimic_mortality"]:
        data_module = TransMedPromptsSTDM(hparams)

    else:
        raise Exception("No dataset available.")

    return data_module
