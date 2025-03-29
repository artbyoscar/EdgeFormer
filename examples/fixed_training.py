import torch
import argparse
import os
import logging
import sys
from torch.utils.data import DataLoader
sys.path.append('.')  # Add the current directory to path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('edgeformer')

def main():
    parser = argparse.ArgumentParser(description="Improved EdgeFormer training script")
    parser.add_argument("--dataset_file", type=str, required=True, help="Path to dataset file")
    parser.add_argument("--vocab_path", type=str, help="Path to vocabulary file (if different from dataset location)")
    parser.add_argument("--model_dir", type=str, default="checkpoints", help="Directory to save models")
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--save_every", type=int, default=10, help="Save model every N epochs")
    args = parser.parse_args()
    
    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Determine vocabulary path
    vocab_path = args.vocab_path
    if vocab_path is None:
        vocab_dir = os.path.dirname(args.dataset_file)
        vocab_path = os.path.join(vocab_dir, 'vocab.pt')
    
    # Load vocabulary to get vocab_size
    logger.info(f"Loading vocabulary from {vocab_path}")
    vocab = torch.load(vocab_path)
    vocab_size = vocab.get('vocab_size')
    if vocab_size is None:
        logger.error("Could not determine vocabulary size")
        return
    
    logger.info(f"Vocabulary size: {vocab_size}")
    
    # Load dataset
    logger.info(f"Loading dataset from {args.dataset_file}")
    dataset = torch.load(args.dataset_file)
    
    # Check dataset type and structure
    logger.info(f"Dataset type: {type(dataset)}")
    
    # If dataset is a list, it's likely the raw dataset
    # We need to create train/val split and DataLoader objects
    from torch.utils.data import TensorDataset, random_split
    
    train_loader = None
    val_loader = None
    
    if isinstance(dataset, list):
        logger.info("Dataset is a list, creating DataLoader objects")
        
        # Create a Dataset from the loaded data
        from src.utils.text_dataset import TextDataset
        
        # Check if each item in the list is already a dictionary with 'input_ids'
        if isinstance(dataset[0], dict) and 'input_ids' in dataset[0]:
            # Already in the right format
            all_data = dataset
        else:
            # Assume it's just input_ids tensors, convert to dictionaries
            all_data = [{'input_ids': item} for item in dataset]
        
        # Split into train/val (90/10)
        train_size = int(0.9 * len(all_data))
        val_size = len(all_data) - train_size
        
        train_data = all_data[:train_size]
        val_data = all_data[train_size:]
        
        # Create DataLoader objects
        train_loader = DataLoader(
            train_data, 
            batch_size=args.batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_data,
            batch_size=args.batch_size,
            shuffle=False
        )
        
        logger.info(f"Created DataLoader objects with {len(train_data)} training samples and {len(val_data)} validation samples")
    
    if train_loader is None or val_loader is None:
        logger.error("Could not create DataLoader objects from dataset")
        return
    
    # Import necessary modules
    try:
        from src.model.config import EdgeFormerConfig
        from src.model.edgeformer import EdgeFormer
    except ImportError as e:
        logger.error(f"Failed to import modules: {e}")
        return
    
    # Create model
    logger.info("Creating model with configuration:")
    logger.info(f"  Hidden size: {args.hidden_size}")
    logger.info(f"  Number of layers: {args.num_layers}")
    logger.info(f"  Number of heads: {args.num_heads}")
    logger.info(f"  Vocabulary size: {vocab_size}")
    
    config = EdgeFormerConfig(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        attention_type='standard'
    )
    
    model = EdgeFormer(config)
    model.to(args.device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs")
    logger.info(f"Training on {args.device}")
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        total_train_loss = 0
        train_batches = 0
        
        for batch in train_loader:
            # Ensure batch has the expected format
            if isinstance(batch, dict) and 'input_ids' in batch:
                input_ids = batch['input_ids'].to(args.device)
                labels = batch['labels'].to(args.device) if 'labels' in batch else input_ids
            else:
                # If it's just a tensor
                input_ids = batch.to(args.device)
                labels = input_ids
            
            # Forward pass
            outputs = model(input_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            # Calculate loss
            loss_fct = torch.nn.CrossEntropyLoss()
            # Reshape logits to [batch_size * seq_len, vocab_size]
            shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            # Reshape labels to [batch_size * seq_len]
            shift_labels = labels[..., 1:].contiguous().view(-1)
            # Calculate loss
            loss = loss_fct(shift_logits, shift_labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            
            total_train_loss += loss.item()
            train_batches += 1
            
            if train_batches % 10 == 0:
                logger.info(f"Epoch {epoch+1}, Batch {train_batches}, Loss: {loss.item():.4f}")
        
        avg_train_loss = total_train_loss / train_batches
        logger.info(f"Epoch {epoch+1} completed, Average training loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        total_val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Ensure batch has the expected format
                if isinstance(batch, dict) and 'input_ids' in batch:
                    input_ids = batch['input_ids'].to(args.device)
                    labels = batch['labels'].to(args.device) if 'labels' in batch else input_ids
                else:
                    # If it's just a tensor
                    input_ids = batch.to(args.device)
                    labels = input_ids
                
                # Forward pass
                outputs = model(input_ids)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
                # Calculate loss
                loss_fct = torch.nn.CrossEntropyLoss()
                # Reshape logits to [batch_size * seq_len, vocab_size]
                shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
                # Reshape labels to [batch_size * seq_len]
                shift_labels = labels[..., 1:].contiguous().view(-1)
                # Calculate loss
                loss = loss_fct(shift_logits, shift_labels)
                
                total_val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = total_val_loss / val_batches
        logger.info(f"Validation loss: {avg_val_loss:.4f}")
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Save model if it's the best so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            logger.info(f"New best validation loss: {best_val_loss:.4f}")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config
            }, os.path.join(args.model_dir, 'best_model.pt'))
        
        # Save model at regular intervals
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config
            }, os.path.join(args.model_dir, f'model_epoch_{epoch+1}.pt'))
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'config': config
    }, os.path.join(args.model_dir, 'final_model.pt'))
    
    logger.info("Training completed!")
    
    # Test generation
    logger.info("Testing text generation...")
    model.eval()
    
    # Get the char_to_idx mapping from vocab
    char_to_idx = vocab['char_to_idx']
    idx_to_char = vocab['idx_to_char']
    
    # Create a simple prompt
    prompt = "EdgeFormer is"
    prompt_ids = [char_to_idx.get(char, vocab_size-1) for char in prompt]
    input_ids = torch.tensor([prompt_ids]).to(args.device)
    
    try:
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                max_length=50,
                temperature=0.7,
                top_k=20
            )
        
        # Decode the generated text
        generated_text = ''.join([idx_to_char.get(id.item(), '[UNK]') for id in generated_ids[0]])
        logger.info(f"Generated text:\n{generated_text}")
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()