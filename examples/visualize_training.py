# Create a new file: examples/visualize_training.py
import matplotlib.pyplot as plt
import argparse
import json
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Visualize training progress")
    parser.add_argument("--log_file", type=str, required=True, help="Path to training log file")
    parser.add_argument("--output_dir", type=str, default="plots", help="Output directory for plots")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load training log
    with open(args.log_file, 'r') as f:
        logs = [json.loads(line) for line in f if line.strip()]
    
    # Extract metrics
    epochs = [log.get('epoch', i) for i, log in enumerate(logs) if 'train_loss' in log]
    train_losses = [log['train_loss'] for log in logs if 'train_loss' in log]
    val_losses = [log['val_loss'] for log in logs if 'val_loss' in log]
    
    # Create loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'loss_curve.png'))
    
    # Create loss ratio plot (validation/training)
    loss_ratios = [val / train for val, train in zip(val_losses, train_losses)]
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_ratios, 'g-')
    plt.xlabel('Epoch')
    plt.ylabel('Validation/Training Loss Ratio')
    plt.title('Validation to Training Loss Ratio')
    plt.axhline(y=1.0, color='r', linestyle='--')
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'loss_ratio.png'))
    
    print(f"Plots saved to {args.output_dir}")
    
    # Print training statistics
    print("\nTraining Statistics:")
    print(f"Initial training loss: {train_losses[0]:.4f}")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Best training loss: {min(train_losses):.4f} (epoch {epochs[np.argmin(train_losses)]})")
    
    print(f"Initial validation loss: {val_losses[0]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    print(f"Best validation loss: {min(val_losses):.4f} (epoch {epochs[np.argmin(val_losses)]})")
    
    # Convergence analysis
    if len(train_losses) >= 3:
        recent_improvement = (train_losses[-3] - train_losses[-1]) / train_losses[-3] * 100
        print(f"\nRecent improvement (last 3 epochs): {recent_improvement:.2f}%")
        
        if recent_improvement < 1.0:
            print("Warning: Training appears to be plateauing (improvement < 1%).")
            print("Consider increasing learning rate or using learning rate scheduling.")

if __name__ == "__main__":
    main()