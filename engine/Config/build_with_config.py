from .config import Config
import torch
import torch.nn as nn
from typing import Optional, List
from ..nnModuleUtil import Classifier, extend_module
from ..Hook import HookBase, LoggerHook, EvalHook, MLFlowLoggerHook
from torch.utils.data import DataLoader
from ..Trainer import Trainer
from typing import Type


## build hooks with config
class HookBuilder:
    def __init__(self, config: Config, trainer: Trainer) -> None:
        self.config = config
        self.trainer = trainer
    def __call__(self, key: Type[HookBase], **kwargs) -> HookBase:

        hook_kwargs = self.config.get(f'HOOKBASE_CONFIG.{key}')
        if hook_kwargs is None:
            hook_kwargs = {}
        
        key_name = key.__name__

        assert isinstance(hook_kwargs, dict), f"Hook kwargs for {key} is not a dictionary"
        hook_kwargs.update(kwargs)
        return key(self.trainer, **hook_kwargs)
    
class ConfigBuilder:
    def __init__(self, config: Config):
        self.cfg = config

    def build_model_with_config(self, **kwargs):
        raise NotImplementedError("build_model_with_config is not implemented yet")
    def build_trainer_with_config(self, **kwargs):
        raise NotImplementedError("build_trainer_with_config is not implemented yet")
    def build_train_loader_with_config(self, **kwargs):
        raise NotImplementedError("build_train_loader_with_config is not implemented yet")
    def build_optimizer_with_config(self, **kwargs):
        raise NotImplementedError("build_optimizer_with_config is not implemented yet")
## build logger with config
