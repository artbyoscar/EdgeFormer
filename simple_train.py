import torch
import os
import sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add current directory to path
sys.path.append('.')

# Load dataset
dataset_path = 'data/focused/text_dataset.pt'
dataset = torch.load(dataset_path)

# Load vocabulary
vocab_path = 'data/focused/vocab.pt'
vocab = torch.load(vocab_path)
vocab_size = vocab['vocab_size']
print(f"Vocabulary size: {vocab_size}")

# Process dataset based on structure
if isinstance(dataset, list):
    # Check if tensors or dictionaries
    if len(dataset) > 0:
        if isinstance(dataset[0], torch.Tensor):
            # List of tensors
            tensors = dataset
        elif isinstance(dataset[0], dict) and 'input_ids' in dataset[0]:
            # List of dictionaries with 'input_ids'
            tensors = [item['input_ids'] for item in dataset]
        else:
            raise ValueError("Unexpected dataset item format")
        
        # Create a dataset
        tensor_data = torch.stack(tensors)
        
        # Split into input and target (shift by 1)
        input_data = tensor_data[:, :-1]
        target_data = tensor_data[:, 1:]
        
        print(f"Input data shape: {input_data.shape}")
        print(f"Target data shape: {target_data.shape}")
        
        # Create dataset and dataloader
        tensor_dataset = TensorDataset(input_data, target_data)
        train_size = int(0.9 * len(tensor_dataset))
        val_size = len(tensor_dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            tensor_dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        print(f"Train loader has {len(train_loader)} batches")
        print(f"Val loader has {len(val_loader)} batches")
    else:
        raise ValueError("Dataset is empty")
else:
    raise ValueError("Dataset is not a list")

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=128, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, input_ids):
        # input_ids shape: [batch_size, seq_len]
        embeddings = self.embedding(input_ids)
        # embeddings shape: [batch_size, seq_len, hidden_size]
        lstm_out, _ = self.lstm(embeddings)
        # lstm_out shape: [batch_size, seq_len, hidden_size]
        logits = self.lm_head(lstm_out)
        # logits shape: [batch_size, seq_len, vocab_size]
        return logits
    
    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=None):
        self.eval()
        current_input = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                # Get prediction for next token
                outputs = self(current_input)
                next_token_logits = outputs[:, -1, :] / temperature
                
                # Apply top-k sampling
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample from the distribution
                probs = nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to current input
                current_input = torch.cat([current_input, next_token], dim=1)
                
        return current_input

# Create model, optimizer, and loss function
model = SimpleModel(vocab_size, hidden_size=128, num_layers=4)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 50
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # Training
    model.train()
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Reshape for loss calculation
        outputs = outputs.view(-1, vocab_size)
        targets = targets.view(-1)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")
    
    # Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            
            # Reshape for loss calculation
            outputs = outputs.view(-1, vocab_size)
            targets = targets.view(-1)
            
            loss = criterion(outputs, targets)
            total_val_loss += loss.item()
    
    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")
    
    # Save model if it's the best so far
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        print(f"New best validation loss: {best_val_loss:.4f}")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        }, 'checkpoints/simple_best_model.pt')
    
    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        }, f'checkpoints/simple_model_epoch_{epoch+1}.pt')

# Save final model
torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': avg_train_loss,
    'val_loss': avg_val_loss
}, 'checkpoints/simple_final_model.pt')

print("Training completed!")

# Test generation
char_to_idx = vocab['char_to_idx']
idx_to_char = vocab['idx_to_char']

# Create a simple prompt
prompt = "EdgeFormer is"
prompt_ids = torch.tensor([[char_to_idx.get(char, vocab_size-1) for char in prompt]])

# Generate text
print(f"Generating text from prompt: '{prompt}'")
generated_ids = model.generate(prompt_ids, max_length=100, temperature=0.7, top_k=20)

# Decode
generated_text = ''.join([idx_to_char.get(id.item(), '[UNK]') for id in generated_ids[0]])
print(f"Generated text:\n{generated_text}")