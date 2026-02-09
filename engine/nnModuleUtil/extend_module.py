import torch
import torch.nn as nn
import typing
from torch.utils.data import DataLoader


class extend_module(nn.Module):
    def __init__(self, model: nn.Module):
        super(extend_module, self).__init__()
        self.model = model
    def forward(self, x):
        return self.model(x)
    
    def config_loss(self, loss_function: typing.Callable):
        self.loss_function = loss_function

    def compute_loss(self, batch):
        raise NotImplementedError("compute_loss is not implemented")
    
    def validation_step(self, test_data_loader: DataLoader):
        raise NotImplementedError("validation_step is not implemented")
    
class Classifier(extend_module):
    def __init__(self, model: nn.Module, loss_function: typing.Callable = nn.CrossEntropyLoss()):
        super(Classifier, self).__init__(model)
        self.config_loss(loss_function)
        self.device = next(self.model.parameters()).device

    def compute_loss(self, batch):
        """
        Simple loss for classification task
        """
        X, Y = batch
        X = X.to(self.device)
        Y = Y.to(self.device)

        Y_hat = self(X)
        loss = self.loss_function(Y_hat, Y)

        if not isinstance(loss, dict):
            return {"loss": loss}
        if 'loss' not in loss:
            return loss.update({"loss": sum(loss.values())})
        return loss
    
    def validation_step(self, test_data_loader: DataLoader):
        """
        Return loss and accuracy on the test dataset
        """
        self.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in test_data_loader:
                X, Y = batch
                X = X.to(self.device)
                Y = Y.to(self.device)
                Y_hat = self(X)
                loss_dict = self.loss_function(Y_hat, Y)
                loss = loss_dict['loss']
                
                total_loss += loss.item() * X.size(0)
                _, predicted = torch.max(Y_hat, 1)
                total_correct += (predicted == Y).sum().item()
                total_samples += X.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return {"val_loss": avg_loss, "val_accuracy": accuracy}
    

# # ===================== Dummy Model ===========================
# class SimpleMLP(Classifier):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(SimpleMLP, self).__init__()
#         self.output_size = output_size
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x