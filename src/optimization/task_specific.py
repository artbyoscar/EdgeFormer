# Create: src/optimization/task_specific.py
class TaskSpecificOptimizer:
    """Optimize compression based on downstream task"""
    
    TASK_CONFIGURATIONS = {
        "sentiment_analysis": {
            "preserve_attention": True,
            "embedding_precision": "high",
            "classification_head": "preserve"
        },
        "named_entity_recognition": {
            "preserve_attention": True,
            "token_level_precision": "high",
            "sequence_modeling": "preserve"
        },
        "text_generation": {
            "preserve_embeddings": True,
            "autoregressive_precision": "high",
            "vocabulary_head": "preserve"
        }
    }