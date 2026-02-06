from .config import Config, register_build_with_config
import torch
import torch.nn as nn
from typing import Optional, List
from ..nnModuleUtil import Classifier, extend_module
from ..Hook import HookBase, LoggerHook, EvalHook, MLFlowLoggerHook
from torch.utils.data import DataLoader
from ..Trainer import Trainer

@register_build_with_config(Config)
def build_model_with_config(self, pre_defined_model: Optional[nn.Module]):
    if pre_defined_model is None and \
        (self.get('MODEL.LOAD_PATH') is None or \
        self.get('MODEL.LOAD_PATH') == ''): # 

        raise ValueError("Model is not defined and config does not provide a path to load model")
    
    assert pre_defined_model is not None, "Pre-defined model is None" ## trick the type checker

    model_type = self.get('MODEL.TYPE')
    if model_type == 'CLASSIFIER':
        return Classifier(model=pre_defined_model)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

@register_build_with_config(Config)
def build_trainer_with_config(self, model: extend_module, train_data_loader: DataLoader, optimizer: torch.optim.Optimizer) -> Trainer:
    ## not in the general case, but for now it is ok
    num_epochs = self.get('TRAINER.NUM_EPOCHS')
    if num_epochs is None:
        raise ValueError("Number of epochs is not set")
    return Trainer(model, train_data_loader, optimizer, num_epochs)

## build hooks with config
class HookBuilder:
    hook_registry = {
        'LoggerHook': LoggerHook,
        'EvalHook': EvalHook,
        'MLFlowLoggerHook': MLFlowLoggerHook,
    }

    def __init__(self, config: Config, trainer: Trainer) -> None:
        self.config = config
        self.trainer = trainer
    def __call__(self, key: str, **kwargs) -> HookBase:
        if key not in HookBuilder.hook_registry:
            raise ValueError(f"Unknown hook class: {key}")
        hook_class = HookBuilder.hook_registry[key]
        hook_kwargs = self.config.get(f'HOOKBASE_CONFIG.{key}')
        if hook_kwargs is None:
            hook_kwargs = {}
        assert isinstance(hook_kwargs, dict), f"Hook kwargs for {key} is not a dictionary"
        hook_kwargs.update(kwargs)
        return hook_class(self.trainer, **hook_kwargs)

## build dataset with config
@register_build_with_config(Config)
def build_train_loader_with_config(self):
    pass

## build optimizer with config
@register_build_with_config(Config)
def build_optimizer_with_config(self, model: nn.Module):
    optimizer_name = self.get('OPTIMIZER.NAME')
    learning_rate = self.get('OPTIMIZER.LEARNING_RATE')

    assert learning_rate is not None

    if optimizer_name == "SGD":
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Invalid optimizer name: {optimizer_name}")

## build logger with config
