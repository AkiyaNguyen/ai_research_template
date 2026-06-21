
import typing
from ..nnModuleUtil import extend_module
import torch
from torch.utils.data import DataLoader
from ..Hook import HookBase
from tqdm import tqdm

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
        if len(self.info_storage) == 0:
            raise ValueError("Info storage is empty")
        return self.info_storage[-1]
    def all_info(self) -> typing.List[dict]:
        return self.info_storage

class Trainer:
    def __init__(self, num_epochs: int, **kwargs) -> None:
        self.num_epochs = num_epochs
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.current_epoch = 0
        self._stop_signal = False
        self.hook: typing.List[HookBase] = []
        self.info_storage: InfoStorage = InfoStorage()

    def stop_training(self) -> None:
        self._stop_signal = True
    def _register_hook(self, hook: HookBase) -> None: 
        self.hook.append(hook)
    def _add_info(self, info: dict) -> None:
        self.info_storage.add_to_latest_info(info)
    
    def _start_train_mode(self) -> None:
        raise NotImplementedError("Subclass must implement this method")
    def train(self) -> None:
        self._start_train_mode()
        
        for hook in self.hook:
            hook.before_train()

        start_epoch = self.current_epoch
        for _ in tqdm(range(start_epoch, self.num_epochs)):
            self.info_storage.add_empty_info()
              
            for hook in self.hook:
                hook.before_train_epoch()

            self.run_step_()

            for hook in self.hook:
                hook.after_train_epoch()

            if self._stop_signal:
                break

            self.current_epoch += 1
            
        for hook in self.hook:
            hook.after_train()
            
    def run_step_(self) -> None:
        raise NotImplementedError("Subclass must implement this method")

        