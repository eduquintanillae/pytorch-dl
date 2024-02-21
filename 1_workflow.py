import torch
from torch import nn # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt

weight = 0.7
bias = 0.3

# -- DATA --
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# -- BUILD MODEL --
# Notes: 
# - nn.Module: almost everything in PyTorch is a nn.Module (think of this as neural network lego blocks)
# - torch.randn(): start with random weights and bias (adjusted as the model learns)
# - PyTorch loves float32 by default
# - forward: defines the computation in the model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__() 
        self.weights = nn.Parameter(
            torch.randn(1, dtype=torch.float), requires_grad=True
        )

        self.bias = nn.Parameter(
            torch.randn(1, dtype=torch.float), requires_grad=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.weights * x + self.bias # linear regression formula (y = m*x + b)
    
torch.manual_seed(42)
model_0 = LinearRegressionModel()
print(list(model_0.parameters()))
print(model_0.state_dict())

with torch.inference_mode(): 
    y_preds = model_0(X_test)

   
# -- TRAIN MODEL --
loss_fn = nn.L1Loss() # MAE loss is same as L1Loss
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01) 

# Training loop
# 1. Forward pass
# 2. Calculate loss
# 3. Zero gradients
# 4. Backward pass
# 5. Optimizer step

# Evaluate
# 1. Forward pass
# 2. Calculate loss

epochs = 100
train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
    model_0.train()

    y_pred = model_0(X_train) # 1
    loss = loss_fn(y_pred, y_train) # 2
    optimizer.zero_grad()# 3
    loss.backward() # 4
    optimizer.step() # 5

    model_0.eval()
    with torch.inference_mode():
      test_pred = model_0(X_test) # 1
      test_loss = loss_fn(test_pred, y_test.type(torch.float)) # 2 

      if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")
            
# -- PREDICT --
model_0.eval()

with torch.inference_mode():
  # model_0.to(device)
  # X_test = X_test.to(device)
  y_preds = model_0(X_test)