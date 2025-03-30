# examples/manufacturing/defect_detection_demo.py
import torch
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from src.model.transformer.config import EdgeFormerConfig
from src.model.transformer.base_transformer import EdgeFormerModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("defect_detection_demo")

def generate_synthetic_data(num_samples=100, defect_rate=0.2):
    """Generate synthetic manufacturing data with defects."""
    # Normal product measurements
    normal_mean = [10.0, 5.0, 8.0, 3.0]
    normal_std = [0.1, 0.05, 0.08, 0.03]
    
    # Defective product measurements 
    defect_mean = [9.8, 5.2, 7.9, 3.1]
    defect_std = [0.2, 0.1, 0.15, 0.06]
    
    # Generate data
    num_defects = int(num_samples * defect_rate)
    num_normal = num_samples - num_defects
    
    # Normal products
    normal_data = np.random.normal(
        loc=normal_mean,
        scale=normal_std,
        size=(num_normal, len(normal_mean))
    )
    normal_labels = np.zeros(num_normal)
    
    # Defective products
    defect_data = np.random.normal(
        loc=defect_mean,
        scale=defect_std,
        size=(num_defects, len(defect_mean))
    )
    defect_labels = np.ones(num_defects)
    
    # Combine data
    data = np.vstack([normal_data, defect_data])
    labels = np.hstack([normal_labels, defect_labels])
    
    # Shuffle
    indices = np.random.permutation(num_samples)
    data = data[indices]
    labels = labels[indices]
    
    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

def main():
    parser = argparse.ArgumentParser(description="EdgeFormer Manufacturing Defect Detection Demo")
    parser.add_argument("--attention", type=str, default="standard", choices=["standard", "mla", "gqa", "sliding_window"])
    parser.add_argument("--visualize", action="store_true", help="Visualize attention patterns")
    parser.add_argument("--profile", action="store_true", help="Profile performance")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to generate")
    args = parser.parse_args()
    
    # Generate synthetic data
    logger.info(f"Generating synthetic manufacturing data with {args.num_samples} samples...")
    data, labels = generate_synthetic_data(num_samples=args.num_samples)
    
    # Split into train/test
    split = int(0.8 * len(data))
    train_data, test_data = data[:split], data[split:]
    train_labels, test_labels = labels[:split], labels[split:]
    
    # Set up model for defect detection
    logger.info(f"Initializing EdgeFormer with {args.attention} attention...")
    config = EdgeFormerConfig(
        attention_type=args.attention,
        hidden_size=256,
        num_attention_heads=8,
        num_hidden_layers=4,
        intermediate_size=512,
        max_position_embeddings=128,
    )
    
    model = EdgeFormerModel(config)
    
    # Simple classification head
    classifier = torch.nn.Linear(config.hidden_size, 2)  # Binary classification
    
    # Generate sequence tokens from measurements
    def measurements_to_tokens(measurements):
        # Scale measurements to integer tokens
        tokens = (measurements * 100).long()
        return tokens
    
    # Convert data to token sequences
    train_tokens = measurements_to_tokens(train_data)
    test_tokens = measurements_to_tokens(test_data)
    
    logger.info("Training classifier...")
    # Very basic "training" - just for demo purposes
    optimizer = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(train_tokens)
        logits = classifier(outputs[0][:, 0, :])  # Use first token representation
        loss = criterion(logits, train_labels)
        loss.backward()
        optimizer.step()
        logger.info(f"Epoch {epoch+1}/5, Loss: {loss.item():.4f}")
    
    # Evaluate
    logger.info("Evaluating model...")
    model.eval()
    with torch.no_grad():
        outputs = model(test_tokens)
        logits = classifier(outputs[0][:, 0, :])
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == test_labels).float().mean()
        logger.info(f"Test Accuracy: {accuracy.item():.4f}")
    
    # Visualize attention if requested
    if args.visualize and hasattr(model, "transformer"):
        logger.info("Visualizing attention patterns...")
        # Run forward pass with attention output
        outputs = model(test_tokens, output_attentions=True)
        attention_outputs = outputs[1]  # Tuple of attention maps
        
        # Visualize attention from last layer
        last_layer_attention = attention_outputs[-1][0]  # First sample, last layer
        
        plt.figure(figsize=(10, 8))
        plt.imshow(last_layer_attention[0].detach().numpy(), cmap='viridis')
        plt.colorbar()
        plt.title(f"Attention Map ({args.attention} attention)")
        plt.xlabel("Key Position")
        plt.ylabel("Query Position")
        plt.savefig(f"attention_{args.attention}.png")
        logger.info(f"Attention visualization saved to attention_{args.attention}.png")
    
    # Profile if requested
    if args.profile:
        logger.info("Profiling model performance...")
        import time
        
        # Warm up
        for _ in range(5):
            _ = model(test_tokens)
        
        # Measure inference time
        start_time = time.time()
        num_runs = 100
        for _ in range(num_runs):
            _ = model(test_tokens)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        logger.info(f"Average inference time: {avg_time*1000:.2f} ms")
        
        # Measure memory usage
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main()