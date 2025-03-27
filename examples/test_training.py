# examples/test_training.py

import logging
from transformers import AutoTokenizer
from src.utils.training import create_dataloaders

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("edgeformer")
logger.info("Testing EdgeFormer training...")

# Create tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Create dataloaders
train_dataloader, val_dataloader = create_dataloaders(
    train_file="data/sample.txt",
    val_file="data/sample_quant.txt",  # Using as validation file
    tokenizer=tokenizer,
    batch_size=2,
    max_length=16,  # Much smaller than before
    stride=8        # Much smaller than before
)

logger.info(f"Created train dataloader with {len(train_dataloader)} batches")
if val_dataloader:
    logger.info(f"Created validation dataloader with {len(val_dataloader)} batches")

logger.info("Training test completed successfully!")