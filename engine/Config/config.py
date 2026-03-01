import yaml
from omegaconf import OmegaConf, DictConfig
import logging

class Config:
    def __init__(self, config_file: str = 'detail_config/simple_MNIST_training.yaml'):
        yaml_cfg = OmegaConf.load(config_file)
        cli_cfg = OmegaConf.from_cli()
        self.config = OmegaConf.merge(yaml_cfg, cli_cfg)
        print("--- Current Configuration ---")
        print(OmegaConf.to_yaml(self.config))
        print("----------------------------")
        ## change to read-only mode
        OmegaConf.set_readonly(self.config, True)

    def get(self, key: str, fallback=None):
        """
        example usage: config.get('model.learning_rate', 1e-4)
        """
        val = OmegaConf.select(self.config, key, default=fallback)
        if val is None and fallback is not None:
            logging.warning(f"Key '{key}' not found, falling back to {fallback}")
            return fallback
        return val