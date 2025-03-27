# src.utils.data_augmentation

## Classes

### SlidingWindowDataset

Dataset for sliding window sampling of long documents.

#### Methods

##### `__init__`

Args:
    documents: List of documents (strings or token lists)
    tokenizer: Tokenizer for tokenizing text
    block_size: Size of text blocks to return
    stride: Stride for sliding window
    apply_augmentation: Whether to apply augmentation
    p_augment: Probability of applying augmentation

```python
__init__(self, documents, tokenizer, block_size=128, stride=64, apply_augmentation=False, p_augment=0.5)
```

##### `create_examples`

Create examples using sliding window.

```python
create_examples(self)
```

### TemperatureBasedSampling

Temperature-based sampling for more diverse training examples.

#### Methods

##### `generate_diverse_samples`

Generate diverse samples using temperature-based sampling.

```python
generate_diverse_samples(model, input_ids, num_samples=4, max_length=128, temperature=1.2, top_k=50, top_p=0.95)
```

##### `sample_from_logits`

Sample from logits with temperature, top-k, and nucleus sampling.

```python
sample_from_logits(logits, temperature=1.0, top_k=0, top_p=0.0)
```

### TextAugmentation

Text augmentation techniques for NLP training.

#### Methods

##### `apply_augmentations`

Apply multiple augmentation techniques.

```python
apply_augmentations(tokens, vocab=None, p_del=0.1, p_swap=0.1, p_repl=0.1)
```

##### `random_deletion`

Randomly delete tokens with probability p.

```python
random_deletion(tokens, p=0.1)
```

##### `random_replacement`

Replace tokens with random tokens from vocabulary with probability p.

```python
random_replacement(tokens, vocab, p=0.1)
```

##### `random_swap`

Randomly swap n pairs of tokens.

```python
random_swap(tokens, n=1)
```

