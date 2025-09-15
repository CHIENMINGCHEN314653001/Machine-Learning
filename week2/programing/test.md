
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 設定中文字體和樣式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('default')

# --- 第二部分：定義函數和模型 ---
def runge_function(x):
    return 1 / (1 + 25 * x**2)

def generate_data(n_samples=500):
    x = np.linspace(-1, 1, n_samples)
    y = runge_function(x)
    return x, y

class StandardMLP(nn.Module):
    def __init__(self, hidden_size=128, num_layers=2):
        super(StandardMLP, self).__init__()
        layers = [nn.Linear(1, hidden_size), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        layers.append(nn.Linear(hidden_size, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class FourierFeatureMLP(nn.Module):
    def __init__(self, num_frequencies=25, hidden_size=128):
        super(FourierFeatureMLP, self).__init__()
        self.num_frequencies = num_frequencies
        self.frequencies = torch.arange(0, num_frequencies + 1).float()
        self.main_network = nn.Sequential(
            nn.Linear(2 * (num_frequencies + 1), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        x_expanded = x.unsqueeze(1)
        k_expanded = self.frequencies.unsqueeze(0).to(x.device)
        argument = 2 * torch.pi * k_expanded * x_expanded
        cos_features = torch.cos(argument)
        sin_features = torch.sin(argument)
        fourier_features = torch.cat([cos_features, sin_features], dim=1)
        return self.main_network(fourier_features)

# --- 第三部分：訓練和評估函數 ---
def train_model(model, x_train, y_train, x_val, y_val, epochs=5000, lr=0.001):
    x_train_t = torch.FloatTensor(x_train).unsqueeze(1)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    x_val_t = torch.FloatTensor(x_val).unsqueeze(1)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(x_train_t)
        loss = criterion(predictions, y_train_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_predictions = model(x_val_t)
            val_loss = criterion(val_predictions, y_val_t)
        
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        if epoch % 1000 == 0:
            print(f'Epoch {epoch}: Train Loss = {loss.item():.6f}, Val Loss = {val_loss.item():.6f}')
    
    return train_losses, val_losses

def evaluate_model(model, x_test, y_test):
    x_test_t = torch.FloatTensor(x_test).unsqueeze(1)
    model.eval()
    with torch.no_grad():
        predictions = model(x_test_t)
        predictions = predictions.squeeze().numpy()
    
    mse = mean_squared_error(y_test, predictions)
    max_error = np.max(np.abs(y_test - predictions))
    return predictions, mse, max_error

# --- 第四部分：主執行流程 ---
print("正在生成數據...")
x, y = generate_data(1000)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

print("初始化模型...")
standard_model = StandardMLP(hidden_size=128, num_layers=3)
fourier_model = FourierFeatureMLP(num_frequencies=25, hidden_size=128)

print("訓練標準MLP模型...")
std_train_loss, std_val_loss = train_model(standard_model, x_train, y_train, x_val, y_val, epochs=5000, lr=0.0005)

print("訓練傅立葉特徵MLP模型...")
fourier_train_loss, fourier_val_loss = train_model(fourier_model, x_train, y_train, x_val, y_val, epochs=3000, lr=0.001)

# --- 第五部分：評估和可視化 ---
x_test = np.linspace(-1, 1, 1000)
y_test = runge_function(x_test)

std_pred, std_mse, std_max_err = evaluate_model(standard_model, x_test, y_test)
fourier_pred, fourier_mse, fourier_max_err = evaluate_model(fourier_model, x_test, y_test)

print(f"\n結果比較:")
print(f"標準MLP - MSE: {std_mse:.2e}, 最大誤差: {std_max_err:.2e}")
print(f"傅立葉MLP - MSE: {fourier_mse:.2e}, 最大誤差: {fourier_max_err:.2e}")

# 創建可視化圖形
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# 函數擬合圖
ax1.plot(x_test, y_test, 'b-', label='真實函數', linewidth=3)
ax1.plot(x_test, std_pred, 'r--', label='標準MLP', linewidth=2)
ax1.set_title('標準MLP逼近效果')
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(x_test, y_test, 'b-', label='真實函數', linewidth=3)
ax2.plot(x_test, fourier_pred, 'g--', label='傅立葉MLP', linewidth=2)
ax2.set_title('傅立葉特徵MLP逼近效果')
ax2.set_xlabel('x')
ax2.set_ylabel('f(x)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 損失曲線圖
ax3.semilogy(std_train_loss, 'r-', label='訓練損失', alpha=0.8)
ax3.semilogy(std_val_loss, 'darkred', label='驗證損失', alpha=0.8)
ax3.set_title('標準MLP: 訓練/驗證損失')
ax3.set_xlabel('訓練輪數')
ax3.set_ylabel('MSE損失 (對數尺度)')
ax3.legend()
ax3.grid(True, alpha=0.3)

ax4.semilogy(fourier_train_loss, 'g-', label='訓練損失', alpha=0.8)
ax4.semilogy(fourier_val_loss, 'darkgreen', label='驗證損失', alpha=0.8)
ax4.set_title('傅立葉MLP: 訓練/驗證損失')
ax4.set_xlabel('訓練輪數')
ax4.set_ylabel('MSE損失 (對數尺度)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('assets/images/runge_approximation_results.png', dpi=300, bbox_inches='tight')
plt.show()

# 誤差比較圖
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x_test, np.abs(y_test - std_pred), 'r-', label='標準MLP誤差', alpha=0.8)
plt.plot(x_test, np.abs(y_test - fourier_pred), 'g-', label='傅立葉MLP誤差', alpha=0.8)
plt.title('絕對誤差比較')
plt.xlabel('x')
plt.ylabel('|誤差|')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.subplot(1, 2, 2)
models = ['標準MLP', '傅立葉MLP']
mse_values = [std_mse, fourier_mse]
colors = ['red', 'green']
plt.bar(models, mse_values, color=colors, alpha=0.7)
plt.title('MSE比較 (對數尺度)')
plt.ylabel('均方誤差')
plt.yscale('log')

plt.tight_layout()
plt.savefig('assets/images/error_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
