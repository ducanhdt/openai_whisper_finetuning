import yaml


def load_config_file(path):
    config = yaml.load(open(path, "r"), Loader=yaml.FullLoader)
    return config
