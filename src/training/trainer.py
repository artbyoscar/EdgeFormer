import torch
import logging

def train(model, tokenizer, train_dataset, optimizer, device, epochs=1):
    """Basic training loop for EdgeFormer."""
    logger = logging.getLogger("edgeformer")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for i, batch in enumerate(train_dataset):
            # Prepare inputs
            inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            # Shift for causal language modeling
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = -100  # Ignore last token as there's no next token to predict
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Calculate loss
            logits = outputs["logits"]
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, model.config.vocab_size), labels.view(-1))
            
            # Backward pass
            loss.backward()
            
            # Log gradient norms to detect issues
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            logger.debug(f"Epoch {epoch}, Batch {i}, Gradient norm: {total_norm}")
            
            # Update weights
            optimizer.step()
            
            # Log progress
            total_loss += loss.item()
            if i % 10 == 0:
                logger.info(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}")
                
        # Log epoch summary
        avg_loss = total_loss / len(train_dataset)
        logger.info(f"Epoch {epoch} completed, Average loss: {avg_loss:.4f}")
