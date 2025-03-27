import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import sys
sys.path.append('.')  # Add project root to path

from src.model.transformer import TransformerModel
from src.utils.text_dataset import TextDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('edgeformer')

# Parameters
seq_length = 32
batch_size = 2
epochs = 5
device = torch.device('cpu')
dataset_file = 'data/text_dataset.pt'
vocab_file = 'data/vocab.pt'
checkpoint_dir = 'checkpoints'
attention_type = 'mla'  # or 'standard'

# Create checkpoint directory if it doesn't exist
os.makedirs(checkpoint_dir, exist_ok=True)

# Load dataset and vocabulary
tokenized_data = torch.load(dataset_file)
vocab_info = torch.load(vocab_file)
vocab_size = vocab_info['vocab_size']

# Prepare dataset manually
data_size = len(tokenized_data) - seq_length
train_size = int(0.9 * data_size)
val_size = data_size - train_size

logger.info(f"Creating dataset with {data_size} sequences, {train_size} for training, {val_size} for validation")

# Create model
model = TransformerModel(
    vocab_size=vocab_size,
    d_model=128,
    nhead=4,
    num_layers=2,
    dim_feedforward=512,
    dropout=0.1,
    attention_type=attention_type
).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    
    for i in range(0, train_size, batch_size):
        batch_indices = torch.arange(i, min(i + batch_size, train_size))
        
        inputs = torch.stack([tokenized_data[j:j+seq_length] for j in batch_indices])
        targets = torch.stack([tokenized_data[j+1:j+seq_length+1] for j in batch_indices])
        
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        if i % 20 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}, Batch {i//batch_size}/{train_size//batch_size}, Loss: {loss.item():.4f}")
    
    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pt")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss / (train_size // batch_size),
    }, checkpoint_path)
    
    logger.info(f"Epoch {epoch+1} completed, average loss: {train_loss / (train_size // batch_size):.4f}")
    logger.info(f"Checkpoint saved to {checkpoint_path}")

# Generate some text
logger.info("Generating sample text")
model.eval()

# Map indices to characters for output
idx_to_char = vocab_info['idx_to_char']

# Generate text
start_seq = tokenized_data[:seq_length].to(device)
input_seq = start_seq.clone()
generated_text = ''.join([idx_to_char[idx.item()] for idx in start_seq])

with torch.no_grad():
    for _ in range(100):  # Generate 100 more characters
        output = model(input_seq.unsqueeze(0))
        next_token_logits = output[0, -1, :]
        next_token = torch.multinomial(torch.softmax(next_token_logits, dim=0), 1)
        
        input_seq = torch.cat([input_seq[1:], next_token])
        generated_text += idx_to_char[next_token.item()]

logger.info(f"Generated text: {generated_text}")

# Save final model
final_path = os.path.join(checkpoint_dir, "final_model.pt")
torch.save(model.state_dict(), final_path)
logger.info(f"Final model saved to {final_path}")