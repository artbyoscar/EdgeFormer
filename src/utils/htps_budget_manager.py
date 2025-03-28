import torch
import math
from typing import Dict, List, Optional, Tuple, Union

class HTPSBudgetManager:
    """
    HyperTree-inspired Budget Manager for controlling inference-time compute.
    Provides intelligent token limiting and thinking extension with optimal path selection.
    """
    
    def __init__(
        self,
        max_tokens: int = 2048,
        extension_token: str = "Wait",
        max_extensions: int = 2,
        thinking_extension_factor: float = 1.5,
        device: str = "cpu"
    ):
        """
        Initialize the HyperTree-inspired Budget Manager.
        
        Args:
            max_tokens: Maximum number of tokens to generate before triggering budget constraints
            extension_token: Token to insert for thinking extension
            max_extensions: Maximum number of thinking extensions allowed
            thinking_extension_factor: Factor to extend thinking time when needed
            device: Device to use for computation (cpu or cuda)
        """
        self.max_tokens = max_tokens
        self.extension_token = extension_token
        self.max_extensions = max_extensions
        self.thinking_extension_factor = thinking_extension_factor
        self.device = device
        self.extensions_used = 0
        self.token_count = 0
        self.computation_paths = {}
        self.path_scores = {}
        
    def reset(self):
        """Reset the budget manager state."""
        self.extensions_used = 0
        self.token_count = 0
        self.computation_paths = {}
        self.path_scores = {}
    
    def update_token_count(self, new_tokens: int):
        """Update the token count with newly generated tokens."""
        self.token_count += new_tokens
        
    def should_extend_thinking(self, 
                               current_output: torch.Tensor, 
                               logits: torch.Tensor, 
                               task_complexity: Optional[float] = None) -> bool:
        """
        Determine if thinking should be extended based on output complexity and uncertainty.
        
        Args:
            current_output: Current generated output tokens
            logits: Logits for the next token prediction
            task_complexity: Optional explicit task complexity score (0-1)
            
        Returns:
            bool: True if thinking should be extended
        """
        # Don't extend if we've reached the maximum extensions allowed
        if self.extensions_used >= self.max_extensions:
            return False
            
        # Use provided task complexity or calculate from output
        complexity = task_complexity if task_complexity is not None else self._estimate_complexity(current_output, logits)
        
        # Calculate uncertainty from logits
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        uncertainty = entropy / math.log(probs.size(-1))  # Normalize by max entropy
        
        # High complexity and high uncertainty suggest need for more thinking
        should_extend = complexity > 0.7 and uncertainty > 0.6
        
        return should_extend
    
    def _estimate_complexity(self, current_output: torch.Tensor, logits: torch.Tensor) -> float:
        """
        Estimate the complexity of the current reasoning task.
        Higher values indicate more complex tasks requiring more compute.
        
        Args:
            current_output: Current generated output tokens
            logits: Logits for next token prediction
            
        Returns:
            float: Complexity score between 0 and 1
        """
        # Count reasoning markers in output as proxy for complexity
        # This is simplified and should be adapted to your specific model outputs
        reasoning_markers = ["therefore", "because", "however", "if", "then", "thus", "so"]
        
        # Convert tensor to text first (this is simplified - adapt to your tokenizer)
        # In practice, you'd use your actual tokenizer's decode method
        output_text = "your_tokenizer.decode(current_output)"
        
        # Count markers
        marker_count = sum(1 for marker in reasoning_markers if marker in output_text.lower())
        
        # Normalize by output length with saturation
        complexity = min(1.0, marker_count / 10)
        
        # Also consider entropy of next token distribution
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean().item()
        max_entropy = math.log(probs.size(-1))
        entropy_factor = entropy / max_entropy
        
        # Combine evidence
        complexity = 0.7 * complexity + 0.3 * entropy_factor
        
        return complexity
    
    def extend_thinking(self, tokenizer, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Extend thinking by inserting extension tokens.
        
        Args:
            tokenizer: Tokenizer for encoding extension tokens
            input_ids: Current input tensor
            
        Returns:
            torch.Tensor: Updated input with extension tokens
        """
        # Encode the extension token
        extension_token_ids = tokenizer.encode(self.extension_token, return_tensors="pt").to(self.device)
        
        # Append extension token to input
        extended_input = torch.cat([input_ids, extension_token_ids], dim=-1)
        
        # Update tracking
        self.extensions_used += 1
        
        return extended_input
    
    def select_optimal_path(self, paths: List[Dict], criteria: Optional[str] = "balanced") -> Dict:
        """
        Select the optimal computation path from available alternatives.
        
        Args:
            paths: List of possible computation paths
            criteria: Selection criteria ("speed", "quality", or "balanced")
            
        Returns:
            Dict: Selected optimal path
        """
        if not paths:
            return {}
            
        # Score each path based on criteria
        scores = []
        for i, path in enumerate(paths):
            speed_score = 1.0 - (path.get("tokens", 0) / self.max_tokens)
            quality_score = path.get("confidence", 0.5)
            
            # Combine scores based on criteria
            if criteria == "speed":
                score = 0.8 * speed_score + 0.2 * quality_score
            elif criteria == "quality":
                score = 0.2 * speed_score + 0.8 * quality_score
            else:  # balanced
                score = 0.5 * speed_score + 0.5 * quality_score
                
            scores.append(score)
            self.path_scores[i] = score
            
        # Select path with best score
        best_path_idx = scores.index(max(scores))
        return paths[best_path_idx]
    
    def enforce_budget(self, 
                      tokenizer,
                      input_ids: torch.Tensor, 
                      logits: torch.Tensor,
                      task_complexity: Optional[float] = None) -> Tuple[torch.Tensor, bool]:
        """
        Enforce the computational budget, extending thinking when beneficial.
        
        Args:
            tokenizer: Tokenizer for encoding tokens
            input_ids: Current input ids
            logits: Logits for next token prediction
            task_complexity: Optional explicit task complexity
            
        Returns:
            Tuple[torch.Tensor, bool]: (Updated input ids, whether generation should continue)
        """
        # Update token count
        self.update_token_count(1)  # 1 new token being considered
        
        # Check if we're within budget
        if self.token_count < self.max_tokens:
            # Still within budget, continue normally
            return input_ids, True
            
        # Check if thinking should be extended
        if self.should_extend_thinking(input_ids, logits, task_complexity):
            # Extend thinking with the extension token
            extended_input = self.extend_thinking(tokenizer, input_ids)
            return extended_input, True
            
        # Budget exhausted and no extension granted
        return input_ids, False