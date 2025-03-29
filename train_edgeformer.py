import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.model.config import EdgeFormerConfig
from src.model.edgeformer_adapter import EdgeFormerAdapter

# Load processed dataset
dataset_path = "data/focused/processed_dataset.pt"
sequences = torch.load(dataset_path)
print(f"Loaded dataset with shape: {sequences.shape}")

# Load vocabulary
vocab_path = "data/focused/vocab.pt"
vocab = torch.load(vocab_path)
vocab_size = vocab["vocab_size"]
print(f"Vocabulary size: {vocab_size}")

# Create input-target pairs (shift by 1)
input_data = sequences[:, :-1]  # All but the last token
target_data = sequences[:, 1:]   # All but the first token
print(f"Input shape: {input_data.shape}, Target shape: {target_data.shape}")

# Create dataset and dataloader
tensor_dataset = TensorDataset(input_data, target_data)
train_size = int(0.9 * len(tensor_dataset))
val_size = len(tensor_dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(
    tensor_dataset, [train_size, val_size]
)

batch_size = 16  # Smaller batch for more complex model
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"Train loader has {len(train_loader)} batches")
print(f"Val loader has {len(val_loader)} batches")

# Create EdgeFormer model
config = EdgeFormerConfig(
    vocab_size=vocab_size,
    hidden_size=64,          # Start smaller
    num_hidden_layers=2,      # Fewer layers initially
    num_attention_heads=4,    # Fewer heads initially
    attention_type="standard"  # Use standard attention first
)

# Create model, optimizer, and loss function
model = EdgeFormerAdapter(config)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, "min", factor=0.5, patience=3, verbose=True
)
criterion = nn.CrossEntropyLoss()

# Create checkpoints directory if it doesn't exist
os.makedirs("checkpoints", exist_ok=True)

# Training loop
num_epochs = 50  # Start with fewer epochs
best_val_loss = float("inf")

print(f"Starting EdgeFormer training for {num_epochs} epochs")

for epoch in range(num_epochs):
    # Training
    model.train()
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Reshape for loss calculation
        outputs = outputs.reshape(-1, vocab_size)
        targets = targets.reshape(-1)
        
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")
    
    # Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            
            # Reshape for loss calculation
            outputs = outputs.reshape(-1, vocab_size)
            targets = targets.reshape(-1)
            
            loss = criterion(outputs, targets)
            total_val_loss += loss.item()
    
    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")
    
    # Update learning rate based on validation loss
    scheduler.step(avg_val_loss)
    
    # Save model if it's the best so far
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        print(f"New best validation loss: {best_val_loss:.4f}")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "config": config
        }, "checkpoints/edgeformer_best_model.pt")
    
    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "config": config
        }, f"checkpoints/edgeformer_model_epoch_{epoch+1}.pt")

# Save final model
torch.save({
    "epoch": num_epochs,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "train_loss": avg_train_loss,
    "val_loss": avg_val_loss,
    "config": config
}, "checkpoints/edgeformer_final_model.pt")

print("EdgeFormer training completed!")

# Test generation
char_to_idx = vocab["char_to_idx"]
idx_to_char = vocab["idx_to_char"]

# Create a simple prompt
prompt = "EdgeFormer is"
prompt_ids = torch.tensor([[char_to_idx.get(char, vocab_size-1) for char in prompt]])

# Generate text
print(f"Generating text from prompt: \"{prompt}\"")
generated_ids = model.generate(prompt_ids, max_length=100, temperature=0.7, top_k=5)

# Decode
generated_text = "".join([idx_to_char.get(id.item(), "[UNK]") for id in generated_ids[0]])
print(f"Generated text:\n{generated_text}")