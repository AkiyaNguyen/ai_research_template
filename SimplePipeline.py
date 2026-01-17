import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import Config
from Hook import *
import mlflow

def get_best_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device

if __name__ == "__main__":
    ## mlp for digits classification
    cfg = Config.Config(config_file='Config/detail_config/simple_MNIST_training.yaml')
    simpleMLP = nn.Sequential(
        nn.Linear(28*28, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    model = Config.build_model_with_config(cfg, simpleMLP)
    model.to(get_best_device())

    optimizer = Config.build_optimizer_with_config(cfg, model)

    ## call MNIST dataset digits
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((28, 28)), transforms.Normalize((0.5,), (0.5,)), lambda x: torch.flatten(x)])
    train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

    test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)

    trainer = Config.build_trainer_with_config(cfg, model, train_data_loader, optimizer)

    ## ====== hooks=======
    hook_builder = Config.HookBuilder(cfg, trainer)
    
    ## build hooks with config
    hook_builder('LoggerHook', LOGGER_FILE='logger.json')
    hook_builder('EvalHook', eval_data_loader=test_data_loader)
    hook_builder('MLFlowLoggerHook', logging_fields=['val_loss', 'val_accuracy', 'loss'])
    ## ====== training=======
    trainer.train()