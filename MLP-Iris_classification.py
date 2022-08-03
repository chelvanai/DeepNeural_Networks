import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics

df = pd.read_csv('iris.csv')

x = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

enc = OneHotEncoder()
y = enc.fit_transform(np.expand_dims(df['species'], 1)).toarray()

print(x.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

X_train = torch.from_numpy(X_train.values).float()
X_test = torch.from_numpy(X_test.values).float()
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test).float()


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(4, 100)
        self.fc2 = nn.Linear(100, 200)
        self.fc3 = nn.Linear(200, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


model = Model()
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(2500):
    # convert numpy array to torch Variable
    inputs = Variable(X_train)
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

# test model
with torch.no_grad():
    inputs = Variable(X_test)
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)

labels = Variable(y_test)
_, y_test_ = torch.max(labels.data, 1)

print("Accuracy:", metrics.accuracy_score(y_test_.numpy(), predicted.numpy()))
