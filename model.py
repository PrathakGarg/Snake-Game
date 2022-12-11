import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

if torch.cuda.is_available():
    print("Activating Cuda GPU")
    torch.cuda.set_device(0)


class LinearQNet(nn.Module):
    def __init__(self, input_size, hidden_layer, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_layer)
        self.dropout1 = nn.Dropout(p=0.3)
        self.linear2 = nn.Linear(hidden_layer, hidden_layer)
        self.dropout2 = nn.Dropout(p=0.3)
        self.linear3 = nn.Linear(hidden_layer, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout1(x)
        x = F.relu(self.linear2(x))
        x = self.dropout2(x)
        x = self.linear3(x)

        return x

    def save(self, file_name="model.pth"):
        model_folder_path = os.path.join(os.getcwd(), "./model")
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optim = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 1. Predict Q values with current state
        pred = self.model(state)
        target = pred.clone()

        # 2. q_new = r + gamma * max(next_Q)
        for i in range(len(done)):
            q_new = reward[i]
            if not done[i]:
                q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i]))

            target[i][torch.argmax(action[i]).item()] = q_new

        self.optim.zero_grad()
        loss = self.criterion(target, pred)

        # l2_lambda = 0.001
        # l2_norm = sum(p.pow(2.0).sum() for p in self.model.parameters())
        # loss = loss + l2_lambda * l2_norm

        loss.backward()

        self.optim.step()
