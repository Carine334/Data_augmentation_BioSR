import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define the VAE model
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # Output: (32, 128, 128)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: (64, 64, 64)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 64 * 64, latent_dim * 2)  # Output: (latent_dim * 2)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 64 * 64),  # Input: (latent_dim)
            nn.ReLU(),
            nn.Unflatten(1, (64, 64, 64)),  # Output: (64, 64, 64)
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),  # Output: (32, 128, 128)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1),  # Output: (1, 256, 256)
            nn.Sigmoid()  # Output pixel values in the range [0, 1]
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        h = self.encoder(x)
        mu, logvar = h[:, :latent_dim], h[:, latent_dim:]
        z = self.reparameterize(mu, logvar)
        # Decode
        return self.decoder(z), mu, logvar

# Define the loss function
def vae_loss(recon_x, x, mu, logvar):
    # Crop the input images to match the size of the reconstructed images
    x = x[:, :, :recon_x.size(2), :recon_x.size(3)]
    
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, KLD  # Return both total loss and KL divergence

# Load and preprocess your NumPy dataset
print("Loading data...")
data = np.load('augmented_microtubule_sr_training_data.npz')
X_train = data['Y']  # Use the loaded data as training data

# Reshape and normalize the data
X_train = X_train.reshape(-1, 1, 256, 256)  # Reshape to (17850, 1, 256, 256)
X_train = X_train.astype(np.float32) / 255.0  # Normalize to [0, 1]

# Clamp the data to ensure it's within the valid range
X_train = np.clip(X_train, 0.0, 1.0)  # Clamping to [0, 1]

# Split the data into training and validation sets
X_train, X_val = train_test_split(X_train, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train)
X_val_tensor = torch.tensor(X_val)

print("Data range:", X_train_tensor.min().item(), X_train_tensor.max().item())

# Hyperparameters
latent_dim = 256
num_epochs = 30  # Adjust this as needed
batch_size = 32
learning_rate = 1e-3
patience = 3  # Number of epochs to wait for improvement before stopping

# Create DataLoaders
train_loader = torch.utils.data.DataLoader(X_train_tensor, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(X_val_tensor, batch_size=batch_size, shuffle=False)

# Initialize the model, optimizer
model = VAE(latent_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Load the model and optimizer state if resuming training
model_save_path = 'vae_model.pth'
resume_training = False  # Set to True if you want to resume training

if resume_training:
    # Load the model state
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']  # Get the last completed epoch
    print(f"Resuming training from epoch {start_epoch}.")
else:
    start_epoch = 0  # Start from the beginning

# Lists to store loss values for plotting
loss_values = []
kl_values = []
val_loss_values = []

# Early stopping variables
best_val_loss = float('inf')
epochs_without_improvement = 0

# Training loop
print("Starting training...")
for epoch in range(start_epoch, num_epochs):
    model.train()
    train_loss = 0
    train_kl = 0
    for batch in train_loader:
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch)
        # Clamp the reconstructed batch
        recon_batch = torch.clamp(recon_batch, 0.0, 1.0)
        total_loss, kl_div = vae_loss(recon_batch, batch, mu, logvar)
        total_loss.backward()
        train_loss += total_loss.item()
        train_kl += kl_div.item()
        optimizer.step()
    
    avg_loss = train_loss / len(train_loader.dataset)
    avg_kl = train_kl / len(train_loader.dataset)
    loss_values.append(avg_loss)  # Store the average loss for this epoch
    kl_values.append(avg_kl)  # Store the average KL divergence for this epoch
    print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}, KL: {avg_kl:.4f}')

    # Validation phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for val_batch in val_loader:
            recon_val_batch, mu_val, logvar_val = model(val_batch)
            recon_val_batch = torch.clamp(recon_val_batch, 0.0, 1.0)
            total_val_loss, _ = vae_loss(recon_val_batch, val_batch, mu_val, logvar_val)
            val_loss += total_val_loss.item()
    
    avg_val_loss = val_loss / len(val_loader.dataset)
    val_loss_values.append(avg_val_loss)
    print(f'Validation Loss: {avg_val_loss:.4f}')

    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
        # Save the model state
        torch.save({
            'epoch': epoch + 1,  # Save the next epoch number
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_save_path)
        print("Model improved and saved.")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break  # Stop training if no improvement

# Save the final model state after training
torch.save({
    'epoch': num_epochs,  # Save the total number of epochs
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, model_save_path)
print(f"Final model saved to '{model_save_path}'.")

print("Training completed. Generating images...")

# Generate 5 images for comparison
model.eval()
with torch.no_grad():
    z = torch.randn(5, latent_dim)
    generated_images = model.decoder(z)

# Plot and save the generated images
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    if i < 5:  # First row for generated images
        ax.imshow(generated_images[i][0].numpy(), cmap='viridis')
        ax.set_title(f'Generated {i + 1}')
    else:  # Second row for original images
        ax.imshow(X_train_tensor[i - 5][0], cmap='viridis')
        ax.set_title(f'Original {i - 4}')
    ax.axis('off')

plt.tight_layout()
plt.savefig('saved_images/generated_vs_original.png')
plt.show()
print("Generated images saved as 'generated_vs_original.png'.")

# Plot the loss and KL divergence evolution
plt.figure(figsize=(12, 6))

# Subplot for Loss
plt.subplot(2, 1, 1)
plt.plot(loss_values, label='Training Loss', marker='o', color='blue')
plt.title('Loss Evolution During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(range(len(loss_values)))
plt.legend()
plt.grid()

# Subplot for KL Divergence
plt.subplot(2, 1, 2)
plt.plot(kl_values, label='KL Divergence', marker='x', color='orange')
plt.title('KL Divergence Evolution During Training')
plt.xlabel('Epoch')
plt.ylabel('KL Divergence')
plt.xticks(range(len(kl_values)))
plt.legend()
plt.grid()

# Save the combined plot
plt.tight_layout()
plt.savefig('saved_images/loss_kl_evolution.png')
plt.show()
print("Loss and KL divergence plot saved as 'loss_kl_evolution.png'.")