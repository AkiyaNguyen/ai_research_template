

from __future__ import annotations
from torch.utils.data import DataLoader 
import json
from typing import TYPE_CHECKING, Any, List
import mlflow
from datetime import datetime
import fnmatch



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
        self.logger_file = kwargs['logger_file']
        if self.logger_file is None:
            raise ValueError("Logger file is not set")
    def before_train_epoch(self) -> None:
        # self.trainer.info_storage.add_empty_info()

        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d %H:%M:%S")

        self.trainer.info_storage.add_to_latest_info({
            'epoch/total': f"{self.trainer.current_epoch + 1} / {self.trainer.num_epochs}",
            'time': dt_string
        })

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
    def __init__(self, trainer: Trainer, logging_fields: List[str] = [], experiment_name: str | None = None, **kwargs: Any) -> None:
        super().__init__(trainer)
        self.logging_fields = logging_fields
        self.name = experiment_name
    
    def found_in_logging_fields(self, query: str) -> bool:
        """
        check if the key is in the logging_fields or 
        """
        return any(fnmatch.fnmatch(query, field) for field in self.logging_fields)
    def before_train(self) -> None:
        if self.name is not None:
            mlflow.set_experiment(self.name)
            print(f"experiment set to {self.name}")
        mlflow.start_run()
    def after_train(self) -> None:
        mlflow.end_run()
    def after_train_epoch(self) -> None:

        for key, value in self.trainer.info_storage.latest_info().items():
            if self.found_in_logging_fields(key):
                mlflow.log_metric(key, value, step=self.trainer.current_epoch)
        