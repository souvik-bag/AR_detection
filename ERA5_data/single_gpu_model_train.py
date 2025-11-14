import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import os
import sys
from tqdm import tqdm
import time
import numpy as np

# Add UNet path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
sys.path.append('/home/sbk29/data/github_AR/AR_detection/u_net')
from unet_block import UNet

# ============================================================================
# Dataset Class
# ============================================================================

class ARDataset(Dataset):
    def __init__(self, image_pt, label_pt):
        super().__init__()
        print(f"Loading image data from: {image_pt}")
        self.image_pt = torch.load(image_pt, weights_only=True)
        print(f"Loading label data from: {label_pt}")
        self.label_pt = torch.load(label_pt, weights_only=True)
        print(f"Dataset loaded: {len(self)} samples")
        print(f"Image shape: {self.image_pt.shape}")
        print(f"Label shape: {self.label_pt.shape}")
    
    def __len__(self):
        return self.image_pt.shape[0]
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.image_pt[idx], self.label_pt[idx]
    

#=============================================================================
# Load indices
#=============================================================================

train_indices = np.load('/home/sbk29/data/github_AR/ERA5_data/final_data/train_indices.npy')
test_indices = np.load('/home/sbk29/data/github_AR/ERA5_data/final_data/test_indices.npy')
validation_indices = np.load('/home/sbk29/data/github_AR/ERA5_data/final_data/validation_indices.npy')

# ============================================================================
# Training Function
# ============================================================================

def train_model(
    model,
    train_loader,
    validation_loader,
    criterion,
    optimizer,
    device,
    num_epochs=10,
    save_dir='checkpoints',
    log_interval=10
):
    """
    Train the model on a single GPU.
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Number of epochs
        save_dir: Directory to save checkpoints
        log_interval: Print loss every N batches
    """
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nCheckpoints will be saved to: {save_dir}")
    
    # Training metrics
    best_loss = float('inf')
    train_losses = []
    validation_losses = []
    
    
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    print(f"Device: {device}")
    print(f"Model: {model.__class__.__name__}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Batches per epoch: {len(train_loader)}")
    print("="*60 + "\n")
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start_time = time.time()
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (features, labels) in enumerate(pbar):
            # Move data to device
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track loss
            batch_loss = loss.item()
            epoch_loss += batch_loss
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{batch_loss:.4f}'})
            
            # Periodic logging
            if batch_idx % log_interval == 0 and batch_idx > 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                print(f"\n  Batch [{batch_idx}/{len(train_loader)}] - Loss: {batch_loss:.4f} (Avg: {avg_loss:.4f})")
        
        val_loss = 0
        model.eval()
        with torch.no_grad():        
            for batch_idx, (features, labels) in enumerate(validation_loader):
                features = features.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
            
                logits = model(features)
                val_loss += criterion(logits, labels).item()
                
                
            
        
        # Epoch statistics
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_epoch_val_loss = val_loss/ len(validation_loader)
        validation_losses.append(avg_epoch_val_loss)
        train_losses.append(avg_epoch_loss)
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Training Loss: {avg_epoch_loss:.4f}")
        print(f"  Validation Loss: {avg_epoch_val_loss:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Samples/sec: {len(train_loader.dataset)/epoch_time:.2f}")
        print(f"{'='*60}\n")
        
        # Save checkpoint every epoch
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss,
            'train_losses': train_losses,
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(save_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
        print(f"✓ Saved latest checkpoint to: {latest_path}")
        
        # Save best model
        if avg_epoch_val_loss < best_loss:
            best_loss = avg_epoch_val_loss
            best_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"✓ New best model! Loss: {best_loss:.4f} - Saved to: {best_path}")
        
        # Save periodic checkpoints (every 5 epochs)
        if (epoch + 1) % 5 == 0:
            epoch_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, epoch_path)
            print(f"✓ Saved periodic checkpoint: {epoch_path}")
        
        print()
    
    # Save final model
    final_checkpoint = {
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_loss': avg_epoch_loss,
        'train_losses': train_losses,
        'best_loss': best_loss,
    }
    
    final_path = os.path.join(save_dir, 'final_model.pth')
    torch.save(final_checkpoint, final_path)
    
    # Save just model weights (smaller file)
    weights_path = os.path.join(save_dir, 'model_weights.pth')
    torch.save(model.state_dict(), weights_path)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Final Loss: {avg_epoch_loss:.4f}")
    print(f"Best Loss: {best_loss:.4f}")
    print(f"Final model saved to: {final_path}")
    print(f"Model weights saved to: {weights_path}")
    print("="*60 + "\n")
    
    return train_losses, validation_losses


# ============================================================================
# Main Function
# ============================================================================

def main():
    # Configuration
    CONFIG = {
        'image_pt': '/home/sbk29/data/github_AR/ERA5_data/final_data/input_X_tensor.pt',
        'label_pt': '/home/sbk29/data/github_AR/ERA5_data/final_data/input_Y_tensor.pt',
        'save_dir': '/mnt/pixstor/data/sbk29/github_AR/checkpoints/run1',
        'batch_size': 4,  # Adjust based on GPU memory
        'num_workers': 2,
        'num_epochs': 10,
        'learning_rate': 1e-4,
        'log_interval': 10,
    }
    
    print("\n" + "="*60)
    print("AR Detection Training - Single GPU")
    print("="*60)
    print("Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")
    
    # Set device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires a GPU.")
    
    device = torch.device('cuda:0')
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    
    # Create dataset and dataloader
    print("Loading dataset...")
    dataset = ARDataset(
        image_pt=CONFIG['image_pt'],
        label_pt=CONFIG['label_pt']
    )
    
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    validation_dataset = Subset(dataset, validation_indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=True,
        persistent_workers=True
    )
    
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True,
        persistent_workers=True
    )
    
    print(f"\nDataLoader created:")
    print(f"  Batch size: {CONFIG['batch_size']}")
    print(f"  Num workers: {CONFIG['num_workers']}")
    print(f"  Total batches: {len(train_loader)}\n")
    
    # Create model
    print("Creating model...")
    model = UNet(n_channels=3, n_classes=1)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1e6:.2f} MB (fp32)\n")
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    print(f"Optimizer: Adam (lr={CONFIG['learning_rate']})")
    print(f"Loss function: BCEWithLogitsLoss\n")
    
    # Train the model
    train_losses, validation_losses = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader= validation_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=CONFIG['num_epochs'],
        save_dir=CONFIG['save_dir'],
        log_interval=CONFIG['log_interval']
    )
    
    # Print training summary
    print("\nTraining Loss History:")
    for epoch, (train_loss, validation_loss) in enumerate(zip(train_losses,validation_losses), 1):
        print(f"  Epoch {epoch}:  Train Loss :{train_loss:.4f}, Validation Loss :{validation_loss:.4f}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        raise