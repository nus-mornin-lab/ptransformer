import argparse
import yaml

def get_all_hparams(general_conf_path):
    parser = argparse.ArgumentParser()

    # general arguments
    parser = get_hparams_from_yaml(parser, general_conf_path)
    model_conf_path = parser.parse_known_args()[0].model_conf_path

    # model arguments
    parser = get_hparams_from_yaml(parser, model_conf_path)
    dataset_conf_path = parser.parse_known_args()[0].dataset_conf_path
    
    # dataset arguments
    parser = get_hparams_from_yaml(parser, dataset_conf_path)
 
    args = parser.parse_args()

    return args


def get_hparams_from_yaml(parser, conf_path):

    with open(conf_path, "r") as file:
        configs = yaml.safe_load(file)

    for key, val in configs.items():
        if type(val) is bool:
            parser.add_argument(f'--{key}',
                            action="store_true",
                            default=val)
        else:
            parser.add_argument(f'--{key}',
                            type=type(val),
                            default=val)
    
    return parser
