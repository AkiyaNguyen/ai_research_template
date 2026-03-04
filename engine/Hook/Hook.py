

from __future__ import annotations
from mlflow import data
from torch.utils.data import DataLoader 
import json
from typing import TYPE_CHECKING, Any, List
import mlflow
from datetime import datetime
import fnmatch
import os 
import matplotlib.pyplot as plt
import typing
import dagshub

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
        # makedirs for the dir of the logger_file
        os.makedirs(os.path.dirname(self.logger_file), exist_ok=True)
    def before_train_epoch(self) -> None:
        # self.trainer.info_storage.add_empty_info()

        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d %H:%M:%S")

        self.trainer.info_storage.add_to_latest_info({
            'epoch/total': f"{self.trainer.current_epoch + 1} / {self.trainer.num_epochs}",
            'time': dt_string
        })

    def after_train_epoch(self) -> None:
        
        print("\n--- Training Result at current epoch ---")
        data = self.trainer.info_storage.latest_info()
        print(json.dumps(data, indent=4))

    def after_train(self) -> None:
        all_data = self.trainer.info_storage.all_info()
        with open(self.logger_file, 'w') as f:
            json.dump(all_data, f, indent=4)

class EvalHook(HookBase):
    def __init__(self, trainer: Trainer, eval_data_loader: DataLoader, **kwargs: Any) -> None:
        super().__init__(trainer)
        self.eval_data_loader = eval_data_loader
    def before_train_epoch(self) -> None:
        pass
    def _run_validation(self) -> dict[Any, Any]:
        raise NotImplementedError("EvalHook does not implement _run_validation, please implement it in a subclass")
    
    def after_train_epoch(self) -> None:
        # result = self.trainer.model.validation_step(self.eval_data_loader) ## dict
        result = self._run_validation()
        self.trainer.info_storage.add_to_latest_info(result)

class MLFlowLoggerHook(HookBase):
    def __init__(self, trainer: Trainer, dagshub_token: str = '', logging_fields: List[str] = [], dagshub_repo_owner: str | None = None, dagshub_repo_name: str | None = None, experiment_name: str | None = None, dir_save_plot: str = "plots", **kwargs: Any) -> None:
        super().__init__(trainer)   
        self.logging_fields = logging_fields
        self.dagshub_repo_owner = dagshub_repo_owner
        self.dagshub_repo_name = dagshub_repo_name
        self.experiment_name = experiment_name
        self.dir_save_plot = dir_save_plot
        self.dagshub_token = dagshub_token
        os.makedirs(self.dir_save_plot, exist_ok=True)
    
    def found_in_logging_fields(self, query: str) -> bool:
        """
        check if the key is in the logging_fields or 
        """
        return any(fnmatch.fnmatch(query, field) for field in self.logging_fields)
    def _plot_metrics(self) -> None:
        all_history = self.trainer.info_storage.all_info()
        
        metrics_to_plot = [k for k in all_history[0].keys() if self.found_in_logging_fields(k)]

        for metric in metrics_to_plot:
            try:
                values = [step_info[metric] for step_info in all_history if metric in step_info]
                
                plt.figure(figsize=(10, 6))
                plt.plot(values, label=metric)
                plt.title(f"Training History: {metric}")
                plt.xlabel("Epoch")
                plt.ylabel("Value")
                plt.legend()
                plt.grid(True)

                plot_path = os.path.join(self.dir_save_plot, f"{metric}.png")
                plt.savefig(plot_path)
                plt.close()

                mlflow.log_artifact(plot_path)
                print(f"Saved and logged plot: {plot_path}")
            except Exception as e:
                print(f"Could not plot {metric}: {e}")
    def before_train(self) -> None:
        if self.dagshub_repo_owner is None or self.dagshub_repo_name is None:
            raise ValueError("DagsHub repo owner and name must be provided for MLFlowLoggerHook")
        os.environ['DAGSHUB_USER_TOKEN'] = self.dagshub_token
        dagshub.init(repo_owner=self.dagshub_repo_owner, repo_name=self.dagshub_repo_name, mlflow=True)
        if self.experiment_name is not None:
            mlflow.set_experiment(self.experiment_name)
            print(f"experiment set to {self.experiment_name}")
        mlflow.start_run()
    def after_train(self) -> None:
        if self.dir_save_plot is not None:
            self._plot_metrics()
        mlflow.end_run()
    def after_train_epoch(self) -> None:

        for key, value in self.trainer.info_storage.latest_info().items():
            if self.found_in_logging_fields(key):
                mlflow.log_metric(key, value, step=self.trainer.current_epoch)
        

class EarlyStoppingHook(HookBase):
    def __init__(self, trainer: Trainer, patience: int = 10, criteria: str = 'val_dice', min_improvement: float = 1e-4, cmp: typing.Callable = lambda a, b: a > b):
        super().__init__(trainer)
        self.patience = patience
        self.criteria = criteria
        self.min_improvement = min_improvement
        self.cmp = cmp
        self.best_value = None
        self.counter = 0
    def after_train_epoch(self) -> None:
        latest_info = self.trainer.info_storage.latest_info()
        if self.criteria not in latest_info:
            raise ValueError(f"Criteria {self.criteria} is not found in latest info")
        current_value = latest_info[self.criteria]
        if self.best_value is None:
            self.best_value = current_value
            return
        if not self.cmp(current_value, self.best_value + self.min_improvement):
            self.counter += 1
            if self.counter >= self.patience:
                self.trainer.stop_training()
                return
        else:
            self.best_value = current_value
            self.counter = 0
    def after_train(self) -> None:
        if self.counter >= self.patience:
            print(f"after training for {self.trainer.current_epoch + 1} epochs")
            print("The training procedure is stopped by EarlyStoppingHook")
