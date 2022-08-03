import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('boston.csv')
df.head()

x = df.drop(['PRICE'], axis=1)
y = df['PRICE']

scaler1 = StandardScaler()
x = scaler1.fit_transform(x)

scaler2 = StandardScaler()
y = scaler2.fit_transform(np.expand_dims(y.values, axis=1))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)

x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)
y_train = torch.from_numpy(y_train).type(torch.FloatTensor)
y_test = torch.from_numpy(y_test).type(torch.FloatTensor)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(13, 26)
        self.fc2 = nn.Linear(26, 52)
        self.fc3 = nn.Linear(52, 104)
        self.fc4 = nn.Linear(104, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


model = Model()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

for epoch in range(5000):
    # convert numpy array to torch Variable
    inputs = Variable(x_train)
    labels = Variable(y_train)

    # clear gradients w.r.t parameters
    optimizer.zero_grad()

    # forward to get output
    outputs = model(inputs)

    # calculate loss
    loss = criterion(outputs, labels)

    # getting gradients w.r.t parameters
    loss.backward()

    # updating parameters
    optimizer.step()

    if epoch % 100 == 0:
        print('epoch {}, loss {}'.format(epoch, loss))

# Test
model.eval()
with torch.no_grad():
    outputs = model(x_test)
    cost = criterion(outputs, y_test)
    print('test loss: %.4f' % cost.item())
