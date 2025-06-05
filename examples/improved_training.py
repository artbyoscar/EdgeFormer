import torch
import argparse
import os
import logging
import sys
from torch.utils.data import DataLoader

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
    parser.add_argument("--model_dir", type=str, default="checkpoints", help="Directory to save models")
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden size")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.003, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--save_every", type=int, default=10, help="Save model every N epochs")
    args = parser.parse_args()
    
    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Load dataset
    logger.info(f"Loading dataset from {args.dataset_file}")
    dataset = torch.load(args.dataset_file)
    
    # Extract vocab size from dataset if available
    vocab_size = None
    if isinstance(dataset, dict) and 'vocab_size' in dataset:
        vocab_size = dataset['vocab_size']
        logger.info(f"Vocabulary size: {vocab_size}")
    else:
        # Try to load vocab separately
        vocab_dir = os.path.dirname(args.dataset_file)
        vocab_path = os.path.join(vocab_dir, 'vocab.pt')
        if os.path.exists(vocab_path):
            vocab = torch.load(vocab_path)
            if isinstance(vocab, dict):
                vocab_size = len(vocab)
                logger.info(f"Vocabulary size from file: {vocab_size}")
    
    if vocab_size is None:
        logger.error("Could not determine vocabulary size")
        return
    
    # Extract DataLoader objects
    train_loader = None
    val_loader = None
    if isinstance(dataset, dict):
        if 'train_loader' in dataset and 'val_loader' in dataset:
            train_loader = dataset['train_loader']
            val_loader = dataset['val_loader']
        elif 'train_data' in dataset and 'val_data' in dataset:
            # Create DataLoader objects
            train_loader = DataLoader(
                dataset['train_data'],
                batch_size=args.batch_size,
                shuffle=True
            )
            val_loader = DataLoader(
                dataset['val_data'],
                batch_size=args.batch_size,
                shuffle=False
            )
    
    if train_loader is None or val_loader is None:
        logger.error("Could not extract data loaders from dataset")
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
            # Ensure batch is on the correct device
            input_ids = batch['input_ids'].to(args.device)
            labels = batch['labels'].to(args.device) if 'labels' in batch else input_ids
            
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
                # Ensure batch is on the correct device
                input_ids = batch['input_ids'].to(args.device)
                labels = batch['labels'].to(args.device) if 'labels' in batch else input_ids
                
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
    
    # Get first few tokens from the dataset
    sample_batch = next(iter(train_loader))
    input_ids = sample_batch['input_ids'][0][:5].unsqueeze(0).to(args.device)
    
    try:
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                max_length=50,
                temperature=0.7,
                top_k=20
            )
        
        # Try to decode the generated text
        # For character-level tokenization
        vocab_dir = os.path.dirname(args.dataset_file)
        vocab_path = os.path.join(vocab_dir, 'vocab.pt')
        if os.path.exists(vocab_path):
            vocab = torch.load(vocab_path)
            if isinstance(vocab, dict):
                id_to_token = {v: k for k, v in vocab.items()}
                generated_text = ''.join([id_to_token.get(id.item(), '[UNK]') for id in generated_ids[0]])
                logger.info(f"Generated text:\n{generated_text}")
    except Exception as e:
        logger.error(f"Error during generation: {e}")

if __name__ == "__main__":
    main()
            # Ensure batch is on the correct device
            input_ids = batch['input_ids'].to(args.device)
            labels = batch['labels'].to(args.device) if 'labels' in batch else input_ids
            
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
                # Ensure batch is on the correct device
                input_ids = batch['input_ids'].to(args.device)
                labels = batch['labels'].to(args.device) if 'labels' in batch else input_ids
                
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
    
    # Get first few tokens from the dataset
    sample_batch = next(iter(train_loader))
    input_ids = sample_batch['input_ids'][0][:5].unsqueeze(0).to(args.device)
    
    try:
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                max_length=50,
                temperature=0.7,
                top_k=20
            )
        
        # Try to decode the generated text
        # For character-level tokenization
        vocab_dir = os.path.dirname(args.dataset_file)
        vocab_path = os.path.join(vocab_dir, 'vocab.pt')
        if os.path.exists(vocab_path):
            vocab = torch.load(vocab_path)
            if isinstance(vocab, dict):
                id_to_token = {v: k for k, v in vocab.items()}
                generated_text = ''.join([id_to_token.get(id.item(), '[UNK]') for id in generated_ids[0]])
                logger.info(f"Generated text:\n{generated_text}")
    except Exception as e:
        logger.error(f"Error during generation: {e}")

if __name__ == "__main__":
    main()
