import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class LoadDataset(Dataset):
    def __init__(self, load, seq_length):
        self.load = load
        self.seq_length = seq_length
        self.data = []
        self.labels = []
        for i in range(len(load) - seq_length):
            self.data.append(load[i:i+seq_length])
            self.labels.append(load[i+seq_length])
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32).to(device)
        y = torch.tensor(self.labels[idx], dtype=torch.float32).to(device)
        return x, y

class TraditionalNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TraditionalNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class BPNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BPNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(1) 
        out, _ = self.lstm(x)
        out = out[:, -1, :]  
        out = self.fc(out)
        return out

def train_model(model, dataloader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}')

def test_model(model, dataloader):
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for inputs, label in dataloader:
            output = model(inputs)
            predictions.extend(output.squeeze().cpu().tolist())
            labels.extend(label.cpu().tolist())
    return predictions, labels

def create_dataloaders(load, seq_length, batch_size=32):
    train_dataset = LoadDataset(load[:int(len(load)*0.8)], seq_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = LoadDataset(load[int(len(load)*0.8):], seq_length)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader

def train_traditional_nn(load, seq_length, hidden_size=64, lr=0.005, epochs=5):
    train_dataloader, test_dataloader = create_dataloaders(load, seq_length)
    model = TraditionalNN(seq_length, hidden_size, 1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_model(model, train_dataloader, criterion, optimizer, epochs)
    predictions, labels = test_model(model, test_dataloader)
    return predictions, labels, model

def train_bp_nn(load, seq_length, hidden_size=64, lr=0.005, epochs=5):
    train_dataloader, test_dataloader = create_dataloaders(load, seq_length)
    model = BPNN(seq_length, hidden_size, 1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_model(model, train_dataloader, criterion, optimizer, epochs)
    predictions, labels = test_model(model, test_dataloader)
    return predictions, labels, model

def train_lstm(load, seq_length, hidden_size=64, lr=0.001, epochs=70):
    train_dataloader, test_dataloader = create_dataloaders(load, seq_length)
    model = LSTMNet(seq_length, hidden_size, 1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_model(model, train_dataloader, criterion, optimizer, epochs)
    predictions, labels = test_model(model, test_dataloader)
    return predictions, labels, model

def train_pso_lstm(load, seq_length, hidden_size, learning_rate, epochs=100):
    train_dataloader, test_dataloader = create_dataloaders(load, seq_length)
    model = LSTMNet(seq_length, hidden_size, 1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_model(model, train_dataloader, criterion, optimizer, epochs)
    predictions, labels = test_model(model, test_dataloader)
    return predictions, labels, model

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x) 