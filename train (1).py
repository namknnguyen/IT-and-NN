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
import matplotlib.pyplot as plt # Added for plotting

# Set random seed for reproducibility
SEED = 1
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED) # For CUDA reproducibility

# Define classes and data sizes
CLASSES = [0, 1, 2, 3, 4]
DATA_SIZES = {'train': 2000, 'val': 250, 'test': 250}

# Define image transformation with normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load and filter training dataset
full_train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_labels = np.array(full_train_dataset.targets)
train_indices = np.where(np.isin(train_labels, CLASSES))[0]
selected_train_indices = np.random.choice(train_indices, size=2250, replace=False)
train_idx = selected_train_indices[:2000]
val_idx = selected_train_indices[2000:]
train_dataset = Subset(full_train_dataset, train_idx)
val_dataset = Subset(full_train_dataset, val_idx)

# Load and filter test dataset
full_test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_labels = np.array(full_test_dataset.targets)
test_indices = np.where(np.isin(test_labels, CLASSES))[0]
selected_test_indices = np.random.choice(test_indices, size=250, replace=False)
test_dataset = Subset(full_test_dataset, selected_test_indices)

# Create DataLoaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Verify dataset sizes
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# --- Global Hyperparameters ---
LEARNING_RATE = 1e-3
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRAD_CLIP_NORM = 1.0              # Gradient clipping norm
PATIENCE = 5                      # Early stopping patience
HIDDEN_SIZE = 512                 # Hidden layer size for the MLP

# New Hyperparameter: Select which knob to vary
# Options: 'KL_divergence', 'latent_entropy', 'effective_bits', 'jacobian_norm_complexity', 'predictive_entropy', 'latent_sparsity', or 'all'
SELECTED_KNOB_TYPE = 'all' # Change this to select a specific knob or 'all'

# --- Knob values ---
KNOBS = {
    'KL_divergence': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],
    'latent_entropy': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],
    # Effective Number of Bits: `knob_value` represents the number of quantization levels.
    # Higher value = more bits = less aggressive quantization.
    # A value of 256 means effectively no quantization (8-bit like).
    'effective_bits': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    # Jacobian Norm Complexity: `knob_value` represents the regularization strength (alpha)
    # Expected: as knob_value increases, final_IT_value (JNC) should decrease.
    'jacobian_norm_complexity': [0.5e-2, 1e-2, 1.5e-2, 2e-2, 2.5e-2, 3e-2, 3.5e-2, 4e-2, 4.5e-2, 5e-2, 5.5e-2, 6e-2, 6.5e-2, 7e-2, 7.5e-2, 8e-2, 8.5e-2, 9e-2, 9.5e-2, 1e-1],
    # New: Predictive Entropy: `knob_value` represents the regularization strength (alpha).
    # Higher knob_value implies stronger penalty for high entropy -> model aims for higher confidence (lower entropy).
    # Expected: as knob_value increases, final_IT_value (Predictive Entropy) should decrease.
    'predictive_entropy': [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10],
    # New: Latent Sparsity: `knob_value` represents the regularization strength (alpha).
    # Changed range to be much wider to observe stronger effects.
    # Higher knob_value implies stronger penalty for non-sparse latent means -> model aims for more zeros in mu.
    # Expected: as knob_value increases, final_IT_value (Latent Sparsity) should decrease.
    'latent_sparsity': [7.05, 7.1, 7.15, 7.2, 7.25, 7.3, 7.35, 7.4, 7.45, 7.5, 7.55, 7.6, 7.65, 7.7, 7.75, 7.8, 7.85, 7.9, 7.95, 8]
}

# --- Model Definition ---
class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=512, num_classes=5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, 128) # Latent dim remains 128
        self.fc_log_var = nn.Linear(hidden_size, 128) # Latent dim remains 128
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x, deterministic=False):
        """
        Added a 'deterministic' flag to bypass random sampling.
        This is crucial for the Jacobian calculation to be consistent.
        """
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)

        if deterministic:
            # Use the mean of the latent distribution for deterministic pass
            h = mu
        else:
            # Use the reparameterization trick for stochastic pass
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            h = mu + eps * std

        logits = self.fc3(h)
        return logits, mu, log_var

# --- Information Theory Quantity Functions ---

def compute_KL_divergence_term(mu, log_var):
    """Computes the raw KL divergence."""
    return 0.5 * (torch.exp(log_var) + mu**2 - 1 - log_var).sum(dim=1).mean()

def compute_latent_entropy_term(log_var, latent_dim=128):
    """Computes the raw entropy of the latent distribution (related to information bottleneck)."""
    return 0.5 * (latent_dim * np.log(2 * np.pi * np.e) + log_var.sum(dim=1)).mean()

def quantize_weights(model, num_levels):
    """
    Quantizes all model parameters to a specified number of levels.
    This function modifies the model's parameters in-place.
    """
    # Using 1024 as the effectively "no quantization" level to accommodate higher knob values
    if num_levels >= 1024:
        return

    for param in model.parameters():
        if param.requires_grad: # Only quantize trainable parameters
            param_data = param.data.detach()

            min_val = param_data.min()
            max_val = param_data.max()

            if max_val == min_val: # Avoid division by zero for constant tensors
                continue

            # Scale to [0, num_levels - 1], round, then scale back to original range
            scaled_param = (param_data - min_val) / (max_val - min_val) * (num_levels - 1)
            quantized_param = torch.round(scaled_param)
            dequantized_param = (quantized_param / (num_levels - 1)) * (max_val - min_val) + min_val

            param.data.copy_(dequantized_param)

def compute_jacobian_norm_complexity(model, data):
    """
    Computes the Jacobian Norm Complexity.
    Defined as the sum of squared gradients of a scalarized output (sum of logits)
    with respect to all model parameters, averaged over the batch size.
    """
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

        return squared_grad_sum / data.size(0) # Average by batch size for consistency

    finally:
        for name, param in model.named_parameters():
            if name in original_requires_grad_states:
                param.requires_grad_(original_requires_grad_states[name])

def compute_predictive_entropy(logits):
    """
    Computes the average predictive entropy over a batch of logits.
    H(Y|X) = - sum_k p_k log p_k
    """
    probabilities = torch.softmax(logits, dim=-1)
    # Clamp probabilities to avoid log(0) and ensure numerical stability
    epsilon = 1e-9
    probabilities = torch.clamp(probabilities, epsilon, 1. - epsilon)

    # Calculate entropy for each sample in the batch and then average
    entropy_per_sample = -torch.sum(probabilities * torch.log(probabilities), dim=-1)
    return entropy_per_sample.mean()

def compute_latent_sparsity(mu):
    """
    Computes the average L1 norm of the latent mean vector mu across a batch.
    This encourages sparsity in the latent representation.
    """
    return torch.abs(mu).mean()

# --- Training and Evaluation Function ---

def train_experiment(knob_type, knob_value, train_loader, val_loader, test_loader, epochs=EPOCHS, learning_rate=LEARNING_RATE, hidden_size=HIDDEN_SIZE):
    # Pass hidden_size to the MLP constructor
    model = MLP(num_classes=len(CLASSES), hidden_size=hidden_size).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    print(f"Starting training for {knob_type}={knob_value}, Hidden Size={hidden_size}")

    for epoch in range(epochs):
        model.train() # Set model to training mode
        train_loss_total, train_ce_loss_sum, train_additional_loss_sum, correct, total = 0, 0, 0, 0, 0

        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
            data, target = data.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad() # Zero gradients for each batch

            # Perform stochastic forward pass for training
            logits, mu, log_var = model(data, deterministic=False)

            ce_loss = criterion(logits, target) # Calculate Cross-Entropy Loss
            additional_loss_term = torch.tensor(0.0, device=DEVICE) # Initialize as a tensor

            if knob_type == 'KL_divergence':
                additional_loss_term = knob_value * compute_KL_divergence_term(mu, log_var)
            elif knob_type == 'latent_entropy':
                additional_loss_term = -knob_value * compute_latent_entropy_term(log_var)
            elif knob_type == 'jacobian_norm_complexity':
                additional_loss_term = knob_value * compute_jacobian_norm_complexity(model, data)
            elif knob_type == 'predictive_entropy':
                # Penalize high predictive entropy: higher knob_value -> lower entropy
                additional_loss_term = knob_value * compute_predictive_entropy(logits)
            elif knob_type == 'latent_sparsity': # New knob: latent_sparsity
                additional_loss_term = knob_value * compute_latent_sparsity(mu)

            total_loss = ce_loss + additional_loss_term
            total_loss.backward() # Backpropagate the total loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM) # Clip gradients to prevent exploding
            optimizer.step() # Update model parameters

            # Apply quantization *after* optimizer step if 'effective_bits' knob is active
            if knob_type == 'effective_bits':
                with torch.no_grad(): # Quantization itself does not need gradients tracked
                    quantize_weights(model, knob_value)

            # Accumulate loss components for logging
            train_loss_total += total_loss.item()
            train_ce_loss_sum += ce_loss.item()
            train_additional_loss_sum += additional_loss_term.item()

            # Calculate training accuracy
            pred = logits.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

        train_loss_avg = train_loss_total / len(train_loader)
        train_accuracy = correct / total

        # --- Validation Phase ---
        model.eval() # Set model to evaluation mode (e.g., disables dropout)
        val_loss = 0
        with torch.no_grad(): # No gradients needed for validation inference
            for data, target in val_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                logits, mu, log_var = model(data, deterministic=False)
                ce_loss_val = criterion(logits, target)
                val_loss += ce_loss_val.item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}, Train Total Loss: {train_loss_avg:.4f} (CE: {train_ce_loss_sum / len(train_loader):.4f}, Add: {train_additional_loss_sum / len(train_loader):.4f}), Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}")

        # --- Early Stopping ---
        if val_loss < best_val_loss:
            best_val_loss, patience_counter = val_loss, 0
            best_model_state = model.state_dict() # Save model state
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    # Load the best performing model based on validation loss before final test
    if best_model_state:
        model.load_state_dict(best_model_state)

    # --- Final Evaluation on Test Set ---
    model.eval() # Ensure model is in evaluation mode for final metrics
    KL_divergence_metric_sum, latent_entropy_metric_sum, jacobian_norm_complexity_sum, predictive_entropy_sum, latent_sparsity_sum = 0.0, 0.0, 0.0, 0.0, 0.0
    correct, total = 0, 0

    for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc="Testing")):
        data, target = data.to(DEVICE), target.to(DEVICE)

        # (A) Calculate Accuracy and other terms within no_grad for efficiency
        with torch.no_grad():
            # Use stochastic pass for accuracy calculation as done during training
            logits, mu, log_var = model(data, deterministic=False)
            pred = logits.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

            if knob_type == 'KL_divergence':
                KL_divergence_metric_sum += compute_KL_divergence_term(mu, log_var).item()
            elif knob_type == 'latent_entropy':
                latent_entropy_metric_sum += compute_latent_entropy_term(log_var).item()
            elif knob_type == 'predictive_entropy':
                # Measure predictive entropy without requiring gradients for evaluation
                predictive_entropy_sum += compute_predictive_entropy(logits).item()
            elif knob_type == 'latent_sparsity': # New knob: latent_sparsity
                latent_sparsity_sum += compute_latent_sparsity(mu).item()

        # (B) Calculate Jacobian Norm Complexity: This metric *requires* gradient computation.
        # It MUST be performed outside `torch.no_grad()` and within `torch.enable_grad()`.
        if knob_type == 'jacobian_norm_complexity':
            with torch.enable_grad(): # Explicitly enable gradient tracking for this block
                jnc_batch_val = compute_jacobian_norm_complexity(model, data).item()
                jacobian_norm_complexity_sum += jnc_batch_val

    test_accuracy = correct / total
    final_it_value = 0.0 # Default value

    # Assign the correct final IT value based on the knob type
    if knob_type == 'KL_divergence':
        final_it_value = KL_divergence_metric_sum / len(test_loader)
    elif knob_type == 'latent_entropy':
        final_it_value = latent_entropy_metric_sum / len(test_loader)
    elif knob_type == 'effective_bits':
        # For effective_bits, the knob_value itself is the IT quantity (number of levels)
        final_it_value = float(knob_value)
    elif knob_type == 'jacobian_norm_complexity':
        # Average the accumulated Jacobian Norm Complexity over all test batches
        final_it_value = jacobian_norm_complexity_sum / len(test_loader)
    elif knob_type == 'predictive_entropy':
        # Average the accumulated Predictive Entropy over all test batches
        final_it_value = predictive_entropy_sum / len(test_loader)
    elif knob_type == 'latent_sparsity': # New knob: latent_sparsity
        final_it_value = latent_sparsity_sum / len(test_loader)


    return test_accuracy, final_it_value

# --- Main Experiment Loop ---
results = []

# Determine which knobs to iterate over based on SELECTED_KNOB_TYPE
knobs_to_run = {}
if SELECTED_KNOB_TYPE == 'all':
    knobs_to_run = KNOBS
elif SELECTED_KNOB_TYPE in KNOBS:
    knobs_to_run[SELECTED_KNOB_TYPE] = KNOBS[SELECTED_KNOB_TYPE]
else:
    print(f"Warning: SELECTED_KNOB_TYPE '{SELECTED_KNOB_TYPE}' not found in KNOBS. Defaulting to 'all' knobs.")
    knobs_to_run = KNOBS # Fallback to running all if an invalid type is given

for knob_type in knobs_to_run:
    for knob_value in knobs_to_run[knob_type]:
        print(f"\n{'='*50}\nRunning experiment: Knob Type = '{knob_type}', Knob Value = {knob_value}, Hidden Size = {HIDDEN_SIZE}\n{'='*50}")
        test_accuracy, final_it_value = train_experiment(
            knob_type, knob_value, train_loader, val_loader, test_loader, hidden_size=HIDDEN_SIZE
        )
        results.append({
            'knob_type': knob_type,
            'knob_value': knob_value,
            'hidden_size': HIDDEN_SIZE, # Add hidden size to results
            'final_IT_value': final_it_value,
            'final_accuracy': test_accuracy
        })
        print(f"Experiment finished: Knob Type = '{knob_type}', Knob Value = {knob_value}, Hidden Size = {HIDDEN_SIZE}")
        print(f"Final Test Accuracy: {test_accuracy:.4f}, Final IT Value: {final_it_value:.4f}")

df = pd.DataFrame(results)
# Save results to a CSV file with a distinct name
df.to_csv('experiment_results_with_latent_sparsity.csv', index=False) # Changed filename for clarity
print("\nResults saved to 'experiment_results_with_latent_sparsity.csv'")


# =======================================================================================
# --- MODIFIED SECTION: PLOTTING GRAPHS ---
# =======================================================================================

# A dictionary for more readable plot labels and titles
knob_name_map = {
    'KL_divergence': 'Kullback-Leibler Divergence',
    'latent_entropy': 'Latent Entropy',
    'effective_bits': 'Effective Bits (Quantization Levels)',
    'jacobian_norm_complexity': 'Jacobian Norm Complexity',
    'predictive_entropy': 'Predictive Entropy',
    'latent_sparsity': 'Latent Sparsity (L1 Norm)'
}

# Get the unique knob types that were actually run from the DataFrame
unique_knobs_in_results = df['knob_type'].unique()

print("\n--- Generating Scatter Plots with Line of Best Fit ---")

for knob_type in unique_knobs_in_results:
    # Filter the DataFrame for the current knob type
    knob_df = df[(df['knob_type'] == knob_type)]

    if len(knob_df) < 2:
        print(f"Skipping plot for '{knob_type}': not enough data points to plot a line of best fit meaningfully.")
        continue

    # Get the human-readable name for the knob, or use the raw name as a fallback
    plot_title_name = knob_name_map.get(knob_type, knob_type)

    # Extract x and y values
    x = knob_df['final_IT_value'].values
    y = knob_df['final_accuracy'].values

    # Calculate the coefficients of the line of best fit (linear regression)
    coefficients = np.polyfit(x, y, 1)
    polynomial = np.poly1d(coefficients)
    y_fit = polynomial(x)

    # Create a new figure for each plot to avoid overlap
    plt.figure(figsize=(10, 6))

    # Create the scatter plot
    plt.scatter(x, y, marker='o', label='Data Points')

    # Plot the line of best fit
    plt.plot(x, y_fit, color='red', linestyle='--', label=f'Line of Best Fit (y={coefficients.round(2)[0]}x + {coefficients.round(2)[1]})')

    # Set the title and labels
    plt.title(f'Final Accuracy vs. {plot_title_name}', fontsize=16)
    plt.xlabel(f'Final {plot_title_name} Value', fontsize=12) # Use knob type for x-axis label
    plt.ylabel('Final Accuracy', fontsize=12)

    # Add a legend
    plt.legend()

    # Improve readability
    plt.grid(True)
    plt.tight_layout()

    # Display the plot
    plt.show()

print("\nAll scatter plots with lines of best fit have been generated.")