import os

from mpi4py import MPI
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np

# 初始化 MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import pydevd_pycharm
port_mapping=[57645,57646,57647]
pydevd_pycharm.settrace('localhost', port=port_mapping[rank], stdoutToServer=True, stderrToServer=True)

print(os.getpid())


# 设置参数
batch_size = 64
learning_rate = 0.01
num_epochs = 5
num_clients = size-1  # 除了中心节点外的客户端数量


# 定义模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 加载数据集并划分数据
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

# 将数据集划分为多个子集
data_size = len(train_dataset) // num_clients
subsets = [Subset(train_dataset, list(range(i * data_size, (i + 1) * data_size))) for i in range(num_clients)]


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy

def get_device():
    """
    Determines the best available device to run computations based on the hardware capabilities.
    Returns:
        device (torch.device): The determined best device ('cuda', 'mps', or 'cpu').
    """
    # Check for CUDA GPU availability
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using CUDA: {torch.cuda.get_device_name(torch.cuda.current_device())}')
    # Check for MPS availability (requires PyTorch 1.12 or higher on macOS with Apple Silicon)
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print('Using Apple MPS')
    # Default to CPU if neither CUDA nor MPS are available
    else:
        device = torch.device('cpu')
        print('Using CPU')
    return device

device=get_device()
# 模型训练和聚合
if rank == 0:
    # 中心节点负责模型聚合
    model_queue = []
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(1, num_epochs + 1):
        # 接收客户端模型
        client_models = []
        for client_rank in range(1, num_clients + 1):
            client_model_params = comm.recv(source=client_rank, tag=epoch)
            client_models.append(client_model_params)

        # 聚合模型参数
        global_model_params = {}
        for key in client_models[0].keys():
            global_model_params[key] = sum([client_model[key] for client_model in client_models]) / len(client_models)

        # 将聚合后的模型参数发送给客户端
        for client_rank in range(1, num_clients + 1):
            comm.send(global_model_params, dest=client_rank, tag=epoch)

        # 创建模型并加载聚合后的参数
        global_model = SimpleNN().to(device)
        global_model.load_state_dict(global_model_params)
        test_loss, test_accuracy = test(global_model, device, test_loader)
        print(f'Epoch {epoch}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

else:
    # 客户端节点负责本地训练
    train_loader = DataLoader(subsets[rank - 1], batch_size=batch_size, shuffle=True)
    model = SimpleNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch)

        # 发送本地模型参数到中心节点
        model_params = model.state_dict()
        comm.send(model_params, dest=0, tag=epoch)

        # 接收聚合后的模型参数并更新本地模型
        global_model_params = comm.recv(source=0, tag=epoch)
        model.load_state_dict(global_model_params)
