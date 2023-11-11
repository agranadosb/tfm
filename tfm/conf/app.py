import yaml

from tfm.conf.schema import Configuration


def configure(config_file: str) -> Configuration:
    with open(config_file, "r") as stream:
        config = yaml.safe_load(stream)

    return Configuration(**config)
