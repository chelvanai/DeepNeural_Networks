import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

X = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.float)
y = torch.tensor([2, 4, 6, 8, 10, 12, 14, 16, 18, 20], dtype=torch.float)

print(X.shape)
print(y.shape)


class LinearDataSet(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


LinearData = LinearDataSet(X, y)
LinearDataloader = torch.utils.data.DataLoader(LinearData, batch_size=1, shuffle=True)


# Linear model
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


model = LinearModel()
critiren = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

print(model)

# Training
for epoch in range(200):
    for data in LinearDataloader:
        x, y = data
        y_pred = model(x)
        loss = critiren(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f'epoch: {epoch}, loss: {loss.item():.4f}')

# Prediction
x_test = torch.tensor([15], dtype=torch.float)
y_pred = model(x_test)
print(f'Prediction: {y_pred.item():.4f}')
