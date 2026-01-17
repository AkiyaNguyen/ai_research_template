
import typing
from extend_module import extend_module
import torch
from torch.utils.data import DataLoader
from Hook import HookBase

class InfoStorage:
    def __init__(self) -> None:
        self.info_storage: typing.List[dict] = []
    def add_to_latest_info(self, info: dict) -> None:
        if len(self.info_storage) == 0:
            raise ValueError("Info storage is empty")
        formatted_info = {k: v.item() if hasattr(v, 'item') else v for k, v in info.items()}
        self.info_storage[-1].update(formatted_info)
    def add_empty_info(self) -> None:
        self.info_storage.append({})
    def latest_info(self) -> dict:
        return self.info_storage[-1]
    def all_info(self) -> typing.List[dict]:
        return self.info_storage

class Trainer:
    def __init__(self, model: extend_module, train_data_loader: DataLoader, optimizer: torch.optim.Optimizer, num_epochs: int) -> None:
        self.model = model
        self.train_data_loader = train_data_loader
        self.optimizer = optimizer
        self.hook: typing.List[HookBase] = []
        self.info_storage: InfoStorage = InfoStorage()
        self.num_epochs = num_epochs
        self.current_epoch = 0
    def _register_hook(self, hook: HookBase) -> None: 
        self.hook.append(hook)

    def train(self) -> None:
        self.model.train()
        for hook in self.hook:
            hook.before_train()

        while self.current_epoch < self.num_epochs:
            for hook in self.hook:
                hook.before_train_epoch()

            self.run_step_()

            for hook in self.hook:
                hook.after_train_epoch()
            self.current_epoch += 1
            
        for hook in self.hook:
            hook.after_train()
            
    def run_step_(self) -> None:
        for batch in self.train_data_loader:
            self.optimizer.zero_grad()
            loss_dict = self.model.compute_loss(batch)
            loss_dict['loss'].backward()
            self.optimizer.step()
            ## add to info storage
            self.info_storage.add_to_latest_info(loss_dict)
