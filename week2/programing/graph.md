!pip install torch numpy matplotlib scikit-learn

# --- Import libraries ---
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set Chinese font (removed since we're using English now)
# plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# --- Define Runge function ---
def runge_function(x):
    return 1 / (1 + 25 * x**2)

def generate_data(n_samples=1000):
    x = np.linspace(-1, 1, n_samples)
    y = runge_function(x)
    return x, y

# --- Define neural network models ---
class StandardMLP(nn.Module):
    def __init__(self, hidden_size=128, num_layers=3):
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
        # Fix: from 0 to num_frequencies (inclusive)
        self.frequencies = torch.arange(0, num_frequencies + 1).float()
        
        # Fix: correct input feature dimension calculation
        input_features = 2 * (num_frequencies + 1)  # cos + sin for each frequency
        
        self.main_network = nn.Sequential(
            nn.Linear(input_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        frequencies = self.frequencies.to(x.device)
        
        # Fix: correct dimension calculation
        x_expanded = x.unsqueeze(-1)  # [batch_size, 1] -> [batch_size, 1, 1]
        k_expanded = frequencies.unsqueeze(0).unsqueeze(0)  # [num_freq] -> [1, 1, num_freq]
        
        # Calculate 2πkx
        argument = 2 * torch.pi * k_expanded * x_expanded  # [batch_size, 1, num_freq]
        
        # Calculate cos and sin
        cos_features = torch.cos(argument).squeeze(1)  # [batch_size, num_freq]
        sin_features = torch.sin(argument).squeeze(1)  # [batch_size, num_freq]
        
        # Combine features
        fourier_features = torch.cat([cos_features, sin_features], dim=1)  # [batch_size, 2 * num_freq]
        
        return self.main_network(fourier_features)

# --- Training function ---
def train_model(model, x_train, y_train, x_val, y_val, epochs=2000, lr=0.001, model_name="Model"):
    x_train_t = torch.FloatTensor(x_train).unsqueeze(1)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    x_val_t = torch.FloatTensor(x_val).unsqueeze(1)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    train_losses, val_losses = [], []
    
    print(f"Training {model_name}...")
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
        
        if epoch % 500 == 0:
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

# --- Main program ---
def main():
    print("=== Neural Network Approximation of Runge Function ===")
    print("Generating data...")
    
    # Generate data
    x, y = generate_data(800)  # Reduce data points to avoid memory issues
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # Create models
    print("Initializing models...")
    standard_model = StandardMLP(hidden_size=64, num_layers=2)  # Reduce network size
    fourier_model = FourierFeatureMLP(num_frequencies=15, hidden_size=64)  # Reduce frequency count
    
    # Check model parameters
    print(f"Standard MLP parameters: {sum(p.numel() for p in standard_model.parameters())}")
    print(f"Fourier MLP parameters: {sum(p.numel() for p in fourier_model.parameters())}")
    
    # Train models
    print("\nTraining Standard MLP...")
    std_train_loss, std_val_loss = train_model(standard_model, x_train, y_train, x_val, y_val, 
                                              epochs=1500, lr=0.0005, model_name="Standard MLP")
    
    print("\nTraining Fourier MLP...")
    fourier_train_loss, fourier_val_loss = train_model(fourier_model, x_train, y_train, x_val, y_val, 
                                                      epochs=1000, lr=0.001, model_name="Fourier MLP")
    
    # Evaluate models
    print("\nEvaluating models...")
    x_test = np.linspace(-1, 1, 1000)
    y_test = runge_function(x_test)
    
    std_pred, std_mse, std_max_err = evaluate_model(standard_model, x_test, y_test)
    fourier_pred, fourier_mse, fourier_max_err = evaluate_model(fourier_model, x_test, y_test)
    
    # Output results
    print(f"\n=== Experimental Results ===")
    print(f"Standard MLP - MSE: {std_mse:.2e}, Max Error: {std_max_err:.2e}")
    print(f"Fourier MLP - MSE: {fourier_mse:.2e}, Max Error: {fourier_max_err:.2e}")
    improvement = (std_mse - fourier_mse) / std_mse * 100
    print(f"MSE Improvement: {improvement:.1f}%")
    
    # Generate charts
    print("\nGenerating charts...")
    
    # 1. Function comparison chart
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    ax1.plot(x_test, y_test, 'b-', label='True Function', linewidth=3)
    ax1.plot(x_test, std_pred, 'r--', label='Standard MLP', linewidth=2)
    ax1.set_title('Standard MLP Approximation')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(x_test, y_test, 'b-', label='True Function', linewidth=3)
    ax2.plot(x_test, fourier_pred, 'g--', label='Fourier MLP', linewidth=2)
    ax2.set_title('Fourier Feature MLP Approximation')
    ax2.set_xlabel('x')
    ax2.set_ylabel('f(x)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Loss curves
    ax3.semilogy(std_train_loss, 'r-', label='Training Loss', alpha=0.8)
    ax3.semilogy(std_val_loss, 'darkred', label='Validation Loss', alpha=0.8)
    ax3.set_title('Standard MLP: Training/Validation Loss')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('MSE Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4.semilogy(fourier_train_loss, 'g-', label='Training Loss', alpha=0.8)
    ax4.semilogy(fourier_val_loss, 'darkgreen', label='Validation Loss', alpha=0.8)
    ax4.set_title('Fourier MLP: Training/Validation Loss')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('MSE Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 2. Error analysis chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(x_test, np.abs(y_test - std_pred), 'r-', label='Standard MLP Error', alpha=0.8)
    ax1.plot(x_test, np.abs(y_test - fourier_pred), 'g-', label='Fourier MLP Error', alpha=0.8)
    ax1.set_title('Absolute Error Comparison')
    ax1.set_xlabel('x')
    ax1.set_ylabel('|Error|')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    models = ['Standard MLP', 'Fourier MLP']
    mse_values = [std_mse, fourier_mse]
    colors = ['red', 'green']
    bars = ax2.bar(models, mse_values, color=colors, alpha=0.7)
    ax2.set_title('MSE Comparison')
    ax2.set_ylabel('Mean Squared Error')
    ax2.set_yscale('log')
    
    for i, (bar, value) in enumerate(zip(bars, mse_values)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, 
                f'{value:.1e}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== Experiment Completed ===")

# Run main program
if __name__ == "__main__":
    main()
