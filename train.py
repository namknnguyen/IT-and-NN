import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

SEED = 1
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

CLASSES = [0, 1, 2, 3, 4]
DATA_SIZES = {'train': 2000, 'val': 250, 'test': 250}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

full_train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_labels = np.array(full_train_dataset.targets)
train_indices = np.where(np.isin(train_labels, CLASSES))[0]
selected_train_indices = np.random.choice(train_indices, size=2250, replace=False)
train_idx = selected_train_indices[:2000]
val_idx = selected_train_indices[2000:]
train_dataset = Subset(full_train_dataset, train_idx)
val_dataset = Subset(full_train_dataset, val_idx)

full_test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_labels = np.array(full_test_dataset.targets)
test_indices = np.where(np.isin(test_labels, CLASSES))[0]
selected_test_indices = np.random.choice(test_indices, size=250, replace=False)
test_dataset = Subset(full_test_dataset, selected_test_indices)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

LEARNING_RATE = 1e-3
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRAD_CLIP_NORM = 1.0
PATIENCE = 5
HIDDEN_SIZE = 512

SELECTED_KNOB_TYPE = 'all'

KNOBS = {
    'KL_divergence': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],
    'latent_entropy': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],
    'effective_bits': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    'jacobian_norm_complexity': [0.5e-2, 1e-2, 1.5e-2, 2e-2, 2.5e-2, 3e-2, 3.5e-2, 4e-2, 4.5e-2, 5e-2, 5.5e-2, 6e-2, 6.5e-2, 7e-2, 7.5e-2, 8e-2, 8.5e-2, 9e-2, 9.5e-2, 1e-1],
    'predictive_entropy': [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10],
    'latent_sparsity': [7.05, 7.1, 7.15, 7.2, 7.25, 7.3, 7.35, 7.4, 7.45, 7.5, 7.55, 7.6, 7.65, 7.7, 7.75, 7.8, 7.85, 7.9, 7.95, 8]
}

class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=512, num_classes=5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, 128)
        self.fc_log_var = nn.Linear(hidden_size, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x, deterministic=False):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)

        if deterministic:
            h = mu
        else:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            h = mu + eps * std

        logits = self.fc3(h)
        return logits, mu, log_var

def compute_KL_divergence_term(mu, log_var):
    return 0.5 * (torch.exp(log_var) + mu**2 - 1 - log_var).sum(dim=1).mean()

def compute_latent_entropy_term(log_var, latent_dim=128):
    return 0.5 * (latent_dim * np.log(2 * np.pi * np.e) + log_var.sum(dim=1)).mean()

def quantize_weights(model, num_levels):
    if num_levels >= 1024:
        return

    for param in model.parameters():
        if param.requires_grad:
            param_data = param.data.detach()

            min_val = param_data.min()
            max_val = param_data.max()

            if max_val == min_val:
                continue

            scaled_param = (param_data - min_val) / (max_val - min_val) * (num_levels - 1)
            quantized_param = torch.round(scaled_param)
            dequantized_param = (quantized_param / (num_levels - 1)) * (max_val - min_val) + min_val

            param.data.copy_(dequantized_param)

def compute_jacobian_norm_complexity(model, data):
    original_requires_grad_states = {}
    for name, param in model.named_parameters():
        if param.is_leaf:
            original_requires_grad_states[name] = param.requires_grad
            param.requires_grad_(True)

    try:
        logits, _, _ = model(data, deterministic=True)
        scalar_output = logits.sum()

        grad_params = torch.autograd.grad(
            outputs=scalar_output,
            inputs=list(model.parameters()),
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )

        squared_grad_sum = 0.0
        for grad in grad_params:
            if grad is not None:
                squared_grad_sum += torch.sum(grad**2)

        if not grad_params or all(g is None for g in grad_params):
             return torch.tensor(0.0, device=data.device)

        return squared_grad_sum / data.size(0)

    finally:
        for name, param in model.named_parameters():
            if name in original_requires_grad_states:
                param.requires_grad_(original_requires_grad_states[name])

def compute_predictive_entropy(logits):
    probabilities = torch.softmax(logits, dim=-1)
    epsilon = 1e-9
    probabilities = torch.clamp(probabilities, epsilon, 1. - epsilon)
    entropy_per_sample = -torch.sum(probabilities * torch.log(probabilities), dim=-1)
    return entropy_per_sample.mean()

def compute_latent_sparsity(mu):
    return torch.abs(mu).mean()

def train_experiment(knob_type, knob_value, train_loader, val_loader, test_loader, epochs=EPOCHS, learning_rate=LEARNING_RATE, hidden_size=HIDDEN_SIZE):
    model = MLP(num_classes=len(CLASSES), hidden_size=hidden_size).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    print(f"Starting training for {knob_type}={knob_value}, Hidden Size={hidden_size}")

    for epoch in range(epochs):
        model.train()
        train_loss_total, train_ce_loss_sum, train_additional_loss_sum, correct, total = 0, 0, 0, 0, 0

        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
            data, target = data.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()

            logits, mu, log_var = model(data, deterministic=False)

            ce_loss = criterion(logits, target)
            additional_loss_term = torch.tensor(0.0, device=DEVICE)

            if knob_type == 'KL_divergence':
                additional_loss_term = knob_value * compute_KL_divergence_term(mu, log_var)
            elif knob_type == 'latent_entropy':
                additional_loss_term = -knob_value * compute_latent_entropy_term(log_var)
            elif knob_type == 'jacobian_norm_complexity':
                additional_loss_term = knob_value * compute_jacobian_norm_complexity(model, data)
            elif knob_type == 'predictive_entropy':
                additional_loss_term = knob_value * compute_predictive_entropy(logits)
            elif knob_type == 'latent_sparsity':
                additional_loss_term = knob_value * compute_latent_sparsity(mu)

            total_loss = ce_loss + additional_loss_term
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            if knob_type == 'effective_bits':
                with torch.no_grad():
                    quantize_weights(model, knob_value)

            train_loss_total += total_loss.item()
            train_ce_loss_sum += ce_loss.item()
            train_additional_loss_sum += additional_loss_term.item()

            pred = logits.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

        train_loss_avg = train_loss_total / len(train_loader)
        train_accuracy = correct / total

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                logits, mu, log_var = model(data, deterministic=False)
                ce_loss_val = criterion(logits, target)
                val_loss += ce_loss_val.item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}, Train Total Loss: {train_loss_avg:.4f} (CE: {train_ce_loss_sum / len(train_loader):.4f}, Add: {train_additional_loss_sum / len(train_loader):.4f}), Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss, patience_counter = val_loss, 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)

    model.eval()
    KL_divergence_metric_sum, latent_entropy_metric_sum, jacobian_norm_complexity_sum, predictive_entropy_sum, latent_sparsity_sum = 0.0, 0.0, 0.0, 0.0, 0.0
    correct, total = 0, 0

    for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc="Testing")):
        data, target = data.to(DEVICE), target.to(DEVICE)

        with torch.no_grad():
            logits, mu, log_var = model(data, deterministic=False)
            pred = logits.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

            if knob_type == 'KL_divergence':
                KL_divergence_metric_sum += compute_KL_divergence_term(mu, log_var).item()
            elif knob_type == 'latent_entropy':
                latent_entropy_metric_sum += compute_latent_entropy_term(log_var).item()
            elif knob_type == 'predictive_entropy':
                predictive_entropy_sum += compute_predictive_entropy(logits).item()
            elif knob_type == 'latent_sparsity':
                latent_sparsity_sum += compute_latent_sparsity(mu).item()

        if knob_type == 'jacobian_norm_complexity':
            with torch.enable_grad():
                jnc_batch_val = compute_jacobian_norm_complexity(model, data).item()
                jacobian_norm_complexity_sum += jnc_batch_val

    test_accuracy = correct / total
    final_it_value = 0.0

    if knob_type == 'KL_divergence':
        final_it_value = KL_divergence_metric_sum / len(test_loader)
    elif knob_type == 'latent_entropy':
        final_it_value = latent_entropy_metric_sum / len(test_loader)
    elif knob_type == 'effective_bits':
        final_it_value = float(knob_value)
    elif knob_type == 'jacobian_norm_complexity':
        final_it_value = jacobian_norm_complexity_sum / len(test_loader)
    elif knob_type == 'predictive_entropy':
        final_it_value = predictive_entropy_sum / len(test_loader)
    elif knob_type == 'latent_sparsity':
        final_it_value = latent_sparsity_sum / len(test_loader)

    return test_accuracy, final_it_value

results = []

knobs_to_run = {}
if SELECTED_KNOB_TYPE == 'all':
    knobs_to_run = KNOBS
elif SELECTED_KNOB_TYPE in KNOBS:
    knobs_to_run[SELECTED_KNOB_TYPE] = KNOBS[SELECTED_KNOB_TYPE]
else:
    print(f"Warning: SELECTED_KNOB_TYPE '{SELECTED_KNOB_TYPE}' not found in KNOBS. Defaulting to 'all' knobs.")
    knobs_to_run = KNOBS

for knob_type in knobs_to_run:
    for knob_value in knobs_to_run[knob_type]:
        print(f"\n{'='*50}\nRunning experiment: Knob Type = '{knob_type}', Knob Value = {knob_value}, Hidden Size = {HIDDEN_SIZE}\n{'='*50}")
        test_accuracy, final_it_value = train_experiment(
            knob_type, knob_value, train_loader, val_loader, test_loader, hidden_size=HIDDEN_SIZE
        )
        results.append({
            'knob_type': knob_type,
            'knob_value': knob_value,
            'hidden_size': HIDDEN_SIZE,
            'final_IT_value': final_it_value,
            'final_accuracy': test_accuracy
        })
        print(f"Experiment finished: Knob Type = '{knob_type}', Knob Value = {knob_value}, Hidden Size = {HIDDEN_SIZE}")
        print(f"Final Test Accuracy: {test_accuracy:.4f}, Final IT Value: {final_it_value:.4f}")

df = pd.DataFrame(results)
df.to_csv('experiment_results_with_latent_sparsity.csv', index=False)
print("\nResults saved to 'experiment_results_with_latent_sparsity.csv'")

knob_name_map = {
    'KL_divergence': 'Kullback-Leibler Divergence',
    'latent_entropy': 'Latent Entropy',
    'effective_bits': 'Effective Bits (Quantization Levels)',
    'jacobian_norm_complexity': 'Jacobian Norm Complexity',
    'predictive_entropy': 'Predictive Entropy',
    'latent_sparsity': 'Latent Sparsity (L1 Norm)'
}

unique_knobs_in_results = df['knob_type'].unique()

print("\n--- Generating Scatter Plots with Line of Best Fit ---")

for knob_type in unique_knobs_in_results:
    knob_df = df[(df['knob_type'] == knob_type)]

    if len(knob_df) < 2:
        print(f"Skipping plot for '{knob_type}': not enough data points to plot a line of best fit meaningfully.")
        continue

    plot_title_name = knob_name_map.get(knob_type, knob_type)

    x = knob_df['final_IT_value'].values
    y = knob_df['final_accuracy'].values

    coefficients = np.polyfit(x, y, 1)
    polynomial = np.poly1d(coefficients)
    y_fit = polynomial(x)

    plt.figure(figsize=(10, 6))

    plt.scatter(x, y, marker='o', label='Data Points')

    plt.plot(x, y_fit, color='red', linestyle='--', label=f'Line of Best Fit (y={coefficients.round(2)[0]}x + {coefficients.round(2)[1]})')

    plt.title(f'Final Accuracy vs. {plot_title_name}', fontsize=16)
    plt.xlabel(f'Final {plot_title_name} Value', fontsize=12)
    plt.ylabel('Final Accuracy', fontsize=12)

    plt.legend()

    plt.grid(True)
    plt.tight_layout()

    plt.show()

print("\nAll scatter plots with lines of best fit have been generated.")
