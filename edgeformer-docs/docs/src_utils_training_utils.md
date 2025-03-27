# src.utils.training_utils

## Classes

### TrainingConfig

Configuration for optimized training pipeline.

#### Methods

##### `__init__`

```python
__init__(self, learning_rate=5e-05, weight_decay=0.01, adam_epsilon=1e-08, warmup_steps=0, max_grad_norm=1.0, gradient_accumulation_steps=1, mixed_precision=False, num_train_epochs=3, per_device_train_batch_size=8, logging_steps=50, save_steps=500, eval_steps=100, output_dir='./output')
```

## Functions

### evaluate

Evaluate the model.

```python
evaluate(model, eval_dataset, config, collate_fn=None)
```

### get_optimizer_and_scheduler

Create optimizer and scheduler with appropriate settings.

```python
get_optimizer_and_scheduler(model, config, num_training_steps)
```

### train

Train the model with optimized pipeline including gradient accumulation and mixed precision.

```python
train(model, train_dataset, config, eval_dataset=None, collate_fn=None)
```

