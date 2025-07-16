import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data as data_utils
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split


# -----------------------------
# Standard Feedforward NN Model
# -----------------------------
class FCNN(nn.Module):
    def __init__(self, input_dim):
        super(FCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Input Convex Neural Network
# -----------------------------
class ICNN(nn.Module):
    def __init__(self, input_dim):
        super(ICNN, self).__init__()
        self.flatten = nn.Flatten()

        # First hidden layer
        self.first_hidden_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU()
        )

        # ICNN hidden layers with non-negative W_z
        self.second_layer_linear_prim = nn.Linear(128, 64)
        self.second_layer_linear_skip = nn.Linear(input_dim, 64)
        self.second_layer_act = nn.ReLU()

        self.third_layer_linear_prim = nn.Linear(64, 32)
        self.third_layer_linear_skip = nn.Linear(input_dim, 32)
        self.third_layer_act = nn.ReLU()

        self.fourth_layer_linear_prim = nn.Linear(32, 16)
        self.fourth_layer_linear_skip = nn.Linear(input_dim, 16)
        self.fourth_layer_act = nn.ReLU()

        self.fifth_layer_linear_prim = nn.Linear(16, 8)
        self.fifth_layer_linear_skip = nn.Linear(input_dim, 8)
        self.fifth_layer_act = nn.ReLU()

        self.output_layer_linear_prim = nn.Linear(8, 1)
        self.output_layer_linear_skip = nn.Linear(input_dim, 1)

        # Non-negative weights
        self.nonneg_layers = [layer for name, layer in self.named_modules() if
                              isinstance(layer, nn.Linear) and "prim" in name]

        # Enforce non-negative weights at initialization
        self.clamp_nonneg_weights()

    def clamp_nonneg_weights(self):
        for name, layer in self.named_modules():
            if isinstance(layer, nn.Linear) and "prim" in name:
                layer.weight.data.clamp_(min=0)

    def forward(self, x):
        x = self.flatten(x)
        z1 = self.first_hidden_layer(x)
        z2 = self.second_layer_act(self.second_layer_linear_prim(z1) + self.second_layer_linear_skip(x))
        z3 = self.third_layer_act(self.third_layer_linear_prim(z2) + self.third_layer_linear_skip(x))
        z4 = self.fourth_layer_act(self.fourth_layer_linear_prim(z3) + self.fourth_layer_linear_skip(x))
        z5 = self.fifth_layer_act(self.fifth_layer_linear_prim(z4) + self.fifth_layer_linear_skip(x))
        out = self.output_layer_linear_prim(z5) + self.output_layer_linear_skip(x)
        return out


# -----------------------------
# Training Functions
# -----------------------------

def train(model, dataloader, optimizer, loss_fn, scheduler=None):
    # Prepare model for training
    model.train()
    for X, y in dataloader:
        # Get prediction / loss
        pred = model(X)
        loss = loss_fn(pred, y)
        # Do backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Scheduler step
        if scheduler:
            scheduler.step()
        # Optional clamping (for ICNN)
        if isinstance(model, ICNN):
            model.clamp_nonneg_weights()


def test(model, dataloader, loss_fn):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    return test_loss / len(dataloader)


def train_loop(model, train_loader, test_loader, loss_fn, optimizer, scheduler=None, epochs=1000):
    # Loop through epochs of training / testing
    history = []
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        train(model, train_loader, optimizer, loss_fn, scheduler=scheduler)
        test_loss = test(model, test_loader, loss_fn)
        history.append(test_loss)
    print("Done!")
    return model, history


def print_trained_models_weights(model):
    for name, parameters in model.named_parameters():
        print(f"{name}: {parameters}")


# -----------------------------
# Create Training Data
# -----------------------------

# Generate training data for f(x) = x^2 - alpha x^4 (non-convex)
np.random.seed(0)
torch.manual_seed(0)


def f(x, alpha=0.005):
    return x ** 2 - alpha * x ** 4


n_samples = 1028
X_np = 4 * np.random.randn(n_samples).reshape(-1, 1)
y_np = f(X_np)

X_train = torch.tensor(X_np, dtype=torch.float32)
y_train = torch.tensor(y_np, dtype=torch.float32)

X_test = torch.linspace(-10, 10, 200).reshape(-1, 1)
y_test = f(X_test)

# -----------------------------
# Train Both Models
# -----------------------------

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {'device'} device")

# Initialize models
nn_model = FCNN(input_dim=1)
icnn_model = ICNN(input_dim=1)

# Loss function and optimizer
loss_fn = nn.MSELoss()  # Mean square error
optimizer_nn = optim.Adam(nn_model.parameters(), lr=0.001)
optimizer_icnn = optim.Adam(icnn_model.parameters(), lr=0.001)

# Number of epochs and batch sizes
n_epochs = 1028  # Number of epochs to run
batch_size = 16  # Size of each batch

# Get training / testing data loaders
train_data = data_utils.TensorDataset(X_train, y_train)
train_loader = data_utils.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data = data_utils.TensorDataset(X_test, y_test)
test_loader = data_utils.DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Train models
nn_model, nn_history = train_loop(nn_model, train_loader, test_loader, loss_fn, optimizer_nn, epochs=n_epochs)
icnn_model, icnn_history = train_loop(icnn_model, train_loader, test_loader, loss_fn, optimizer_icnn, epochs=n_epochs)

# -----------------------------
# Plotting
# -----------------------------

x_test = torch.linspace(-10, 10, 200).reshape(-1, 1)
y_true = f(x_test)

with torch.no_grad():
    y_nn_pred = nn_model(x_test)
    y_icnn_pred = icnn_model(x_test)

plt.figure(figsize=(10, 5))
plt.plot(x_test, y_true, label='True $f(x) = x^2 - \\alpha x^4$', color='black')
plt.plot(x_test, y_nn_pred, label='Standard NN', linestyle='--')
plt.plot(x_test, y_icnn_pred, label='ICNN', linestyle='-.')

plt.scatter(X_train, y_train, c='green', s=15, alpha=0.7, label='Training data')
plt.title('Function Approximation')
plt.legend()
plt.grid(True)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()

# -----------------------------
# Training / Testing for IEEE Case Data
# -----------------------------

import pathlib

curr_dir = pathlib.Path(".")
data_dir = curr_dir / "training_data"
num_samples = 1000

### Load IEEE case training data for ICNN

# cases = ["case9", "case14", "case30", "case57", "case118", "case300"]

# Define case name
case_name = "case14"
print("Case: ", case_name)

# Load data
df_data = pd.read_csv(data_dir / f"{case_name}_{num_samples}_samples.csv")

# Get feasible data points only
df_data = df_data.loc[df_data["feas_flag"] == 1]

# Organize data
num_buses = int((len(df_data.columns) - 2) / 2)
print("Number of buses: ", num_buses)
assert int(case_name.strip("case")) == num_buses
X_columns = [f"Pd{i+1}" for i in range(num_buses)] + [f"Qd{i+1}" for i in range(num_buses)]
assert set(X_columns).issubset(set(df_data.columns))
assert "Cost" in df_data.columns
df_X = df_data[X_columns]
df_y = df_data["Cost"]

# Get input data/labels
X, y = df_X.values, df_y.values.reshape(-1, 1)

### Train Neural Network(s)

# Create train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=5)
X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

# Get training / testing data loaders
train_data = data_utils.TensorDataset(X_train, y_train)
train_loader = data_utils.DataLoader(train_data, batch_size=batch_size)
test_data = data_utils.TensorDataset(X_test, y_test)
test_loader = data_utils.DataLoader(test_data, batch_size=batch_size)

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {'device'} device")

# Initialize models
icnn_model = ICNN(input_dim= 2 * num_buses)
nn_model = FCNN(input_dim=2 * num_buses)

# Loss function and optimizer
loss_fn = nn.MSELoss()  # Mean square error
optimizer_icnn = optim.Adam(icnn_model.parameters(), lr=0.001)
optimizer_nn = optim.Adam(nn_model.parameters(), lr=0.001)

# Number of epochs and batch sizes
n_epochs = 2056   # Number of epochs to run
batch_size = 8  # Size of each batch

# Train model(s)
icnn_model, icnn_history = train_loop(icnn_model, train_loader, test_loader, loss_fn, optimizer_icnn, epochs=n_epochs)
nn_model, nn_history = train_loop(nn_model, train_loader, test_loader, loss_fn, optimizer_nn, epochs=n_epochs)

# Test model
plt.title("Test Error (MSE)")
plt.plot(icnn_history, label="ICNN")
plt.plot(nn_history, label="FCNN")
plt.show()