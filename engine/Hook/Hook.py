

from __future__ import annotations
from torch.utils.data import DataLoader 
import json
from typing import TYPE_CHECKING, Any, List
import mlflow

if TYPE_CHECKING:
    from engine.Trainer import Trainer
    

class HookBase:
    def __init__(self, trainer: Trainer) -> None:
        self.trainer = trainer
        self.trainer._register_hook(self)
    
    def before_train(self) -> None:
        pass
    def after_train(self) -> None:
        pass
    def before_train_epoch(self) -> None:
        pass
    def after_train_epoch(self) -> None:
        pass

class LoggerHook(HookBase):
    def __init__(self, trainer: Trainer, **kwargs: Any) -> None:
        super().__init__(trainer)
        self.logger_file = kwargs['LOGGER_FILE']
        if self.logger_file is None:
            raise ValueError("Logger file is not set")
    def before_train_epoch(self) -> None:
        # self.trainer.info_storage.add_empty_info()
        pass
    def after_train_epoch(self) -> None:
        pass

    def after_train(self) -> None:
        with open(self.logger_file, "w") as f:
            json.dump(self.trainer.info_storage.all_info(), f, indent=4)

class EvalHook(HookBase):
    def __init__(self, trainer: Trainer, eval_data_loader: DataLoader, **kwargs: Any) -> None:
        super().__init__(trainer)
        self.eval_data_loader = eval_data_loader
    def before_train_epoch(self) -> None:
        pass
    def after_train_epoch(self) -> None:
        result = self.trainer.model.validation_step(self.eval_data_loader) ## dict
        self.trainer.info_storage.add_to_latest_info(result)

class MLFlowLoggerHook(HookBase):
    def __init__(self, trainer: Trainer, logging_fields: List[str] = [], **kwargs: Any) -> None:
        super().__init__(trainer)
        self.logging_fields = logging_fields
        
    def before_train(self) -> None:
        mlflow.start_run()
    def after_train(self) -> None:
        mlflow.end_run()
    def after_train_epoch(self) -> None:
        for key, value in self.trainer.info_storage.latest_info().items():
            if key in self.logging_fields:
                mlflow.log_metric(key, value, step=self.trainer.current_epoch)
        