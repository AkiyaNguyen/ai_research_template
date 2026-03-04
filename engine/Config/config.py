import yaml
from omegaconf import OmegaConf, DictConfig
import logging

class Config:
    def __init__(self, config_file: str = 'detail_config/simple_MNIST_training.yaml',
            cli_overrides=None):
        yaml_cfg = OmegaConf.load(config_file)
        if cli_overrides is not None:
            cli_cfg = OmegaConf.from_cli(cli_overrides)
        else:
            cli_cfg = OmegaConf.create()
            
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
    
    def set(self, key: str, value):
        OmegaConf.set_readonly(self.config, False)
        try:
            OmegaConf.update(self.config, key, value, merge=True)
        except Exception as e:
            logging.error(f"Failed to update config key '{key}': {e}")
        finally:
            OmegaConf.set_readonly(self.config, True)