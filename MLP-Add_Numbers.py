import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np


# Addition dataset create function
def create_addition_dataset(n_samples=500, max_val=100):
    x = torch.randint(0, max_val, (n_samples, 2))
    y = x[:, 0] + x[:, 1]
    return x, y


# make dataset
x, y = create_addition_dataset()

print(x[0:10], y[0:10])

scaler1 = MinMaxScaler()
x = scaler1.fit_transform(x)

scaler2 = MinMaxScaler()
y = scaler2.fit_transform(np.expand_dims(y, axis=1))


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(2, 100)
        self.fc2 = nn.Linear(100, 200)
        self.fc3 = nn.Linear(200, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


model = Model()
print(model)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

X = torch.from_numpy(x).float()
Y = torch.from_numpy(y).float()

for epoch in range(100):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = criterion(y_pred, Y)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'epoch: {epoch}, loss: {loss.item():.4f}')

# Test
test = np.array([[17., 31.]])

test = (test - scaler1.data_min_) / (scaler1.data_max_ - scaler1.data_min_)

res = model(torch.from_numpy(test).float())

print(scaler2.inverse_transform(np.expand_dims(np.array([res.item()]), axis=1)))
