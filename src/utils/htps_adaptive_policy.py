import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional

class HTPSAdaptivePolicy:
    """
    HyperTree-Enhanced Adaptive Iteration Policy for determining optimal
    iteration counts and computational paths for different task types.
    """
    def __init__(self, config=None):
        # Default configuration
        self.default_min_iterations = 2
        self.default_max_iterations = 16
        self.default_threshold = 0.01
        
        # Task-specific configurations
        self.task_configs = {
            "simple": {"min_iter": 1, "max_iter": 4, "threshold": 0.02},
            "retrieval": {"min_iter": 2, "max_iter": 6, "threshold": 0.015},
            "reasoning": {"min_iter": 4, "max_iter": 32, "threshold": 0.008},
            "math": {"min_iter": 8, "max_iter": 64, "threshold": 0.005},
            "creative": {"min_iter": 3, "max_iter": 12, "threshold": 0.012}
        }
        
        # Import task configs from provided config if available
        if config and hasattr(config, "task_configs"):
            self.task_configs.update(config.task_configs)
            
        # Path selection parameters
        self.exploration_factor = 0.2  # Balance between exploration and exploitation
        self.path_memory_size = 5      # Number of previous paths to remember
        self.path_memory = {}          # Dictionary to store path performance
        
        # Token-specific parameters
        self.token_iteration_history = {}  # Track iteration counts by token type
        self.token_positions_history = {}  # Track iteration counts by position
        
        # Statistics for analysis
        self.iteration_stats = {
            "total_tokens": 0,
            "total_iterations": 0,
            "iterations_by_task": {},
            "iterations_by_position": {},
        }
        
    def detect_task_type(self, prompt: str) -> str:
        """
        Detect the task type based on prompt content.
        
        Args:
            prompt: Input prompt text
            
        Returns:
            Task type as string
        """
        prompt_lower = prompt.lower()
        
        # Check for math-related keywords
        if any(keyword in prompt_lower for keyword in 
               ["solve", "calculate", "equation", "math problem", "arithmetic"]):
            return "math"
            
        # Check for reasoning-related keywords
        if any(keyword in prompt_lower for keyword in 
               ["reason", "explain", "analyze", "deduce", "think step by step"]):
            return "reasoning"
            
        # Check for retrieval-focused tasks
        if any(keyword in prompt_lower for keyword in 
               ["who", "what", "when", "where", "list"]):
            return "retrieval"
            
        # Check for creative tasks
        if any(keyword in prompt_lower for keyword in 
               ["create", "generate", "write", "story", "poem"]):
            return "creative"
            
        # Default to simple
        return "simple"
    
    def get_iteration_params(self, prompt: str = "", task_type: str = None) -> Tuple[int, int, float]:
        """
        Get optimal iteration parameters based on prompt or specified task type.
        
        Args:
            prompt: Input prompt text
            task_type: Optional task type override
            
        Returns:
            Tuple of (min_iterations, max_iterations, convergence_threshold)
        """
        # Detect task type if not provided
        if task_type is None and prompt:
            task_type = self.detect_task_type(prompt)
        elif task_type is None:
            task_type = "simple"  # Default
            
        # Get parameters for task type
        if task_type in self.task_configs:
            params = self.task_configs[task_type]
            return (
                params["min_iter"],
                params["max_iter"],
                params["threshold"]
            )
        
        # Return defaults if task type not recognized
        return (
            self.default_min_iterations,
            self.default_max_iterations,
            self.default_threshold
        )
    
    def select_computation_path(self, value_history: List[float], 
                               token_info: Dict, 
                               available_paths: List[str]) -> str:
        """
        Select optimal computation path based on value history and token info.
        
        Args:
            value_history: History of value estimates
            token_info: Information about the current token
            available_paths: List of available computation paths
            
        Returns:
            Selected path name
        """
        # Default if only one path available
        if len(available_paths) <= 1:
            return available_paths[0] if available_paths else "default"
            
        # Calculate path scores
        path_scores = {}
        for path in available_paths:
            # Base score from past performance if available
            base_score = self.path_memory.get(path, {}).get("avg_value", 0.5)
            
            # Adjust score based on value history trend if available
            trend_factor = 1.0
            if len(value_history) >= 3:
                # Positive trend increases score
                if value_history[-1] > value_history[-3]:
                    trend_factor = 1.2
                # Negative trend decreases score
                elif value_history[-1] < value_history[-3]:
                    trend_factor = 0.8
            
            # Calculate final score with exploration bonus
            exploration_bonus = self.exploration_factor / (self.path_memory.get(path, {}).get("count", 0) + 1)
            path_scores[path] = base_score * trend_factor + exploration_bonus
        
        # Select path with highest score
        selected_path = max(path_scores, key=path_scores.get)
        
        # Update path memory with selection
        if selected_path not in self.path_memory:
            self.path_memory[selected_path] = {"count": 0, "values": [], "avg_value": 0.5}
        self.path_memory[selected_path]["count"] += 1
        
        return selected_path
    
    def update_path_performance(self, path_name: str, value: float):
        """
        Update performance metrics for a computation path.
        
        Args:
            path_name: Name of the path
            value: Performance value (0-1)
        """
        if path_name not in self.path_memory:
            self.path_memory[path_name] = {"count": 0, "values": [], "avg_value": 0.5}
            
        # Update path memory
        path_data = self.path_memory[path_name]
        path_data["values"].append(value)
        path_data["values"] = path_data["values"][-self.path_memory_size:]  # Keep last N values
        path_data["avg_value"] = sum(path_data["values"]) / len(path_data["values"])
    
    def get_dynamic_iterations(self, token_id: int, position: int, 
                              task_type: str, context_value: float) -> int:
        """
        Get dynamic max iterations based on token position and type.
        
        Args:
            token_id: ID of the token
            position: Position in sequence
            task_type: Type of task
            context_value: Value of context so far
            
        Returns:
            Dynamic max iterations
        """
        # Get base parameters
        min_iter, max_iter, _ = self.get_iteration_params(task_type=task_type)
        
        # Position-based adjustments
        position_factor = 1.0
        if position < 10:  # Early tokens often need more iterations
            position_factor = 1.5
        elif position > 100:  # Later tokens can often use fewer iterations
            position_factor = 0.8
            
        # Token-specific adjustments based on history
        token_factor = 1.0
        if str(token_id) in self.token_iteration_history:
            avg_iters = self.token_iteration_history[str(token_id)]["avg_iterations"]
            # If this token historically needs more iterations, increase factor
            if avg_iters > max_iter / 2:
                token_factor = 1.3
            # If this token historically needs fewer iterations, decrease factor
            elif avg_iters < max_iter / 4:
                token_factor = 0.7
                
        # Context value adjustments - lower values may need more iterations
        context_factor = 1.0
        if context_value < 0.3:
            context_factor = 1.5
        elif context_value > 0.8:
            context_factor = 0.7
            
        # Calculate dynamic max iterations
        dynamic_max = int(max_iter * position_factor * token_factor * context_factor)
        
        # Clamp to range
        return max(min_iter, min(max_iter, dynamic_max))
    
    def record_token_iterations(self, token_id: int, position: int, 
                               iterations_used: int, task_type: str):
        """
        Record iterations used for a token for future reference.
        
        Args:
            token_id: ID of the token
            position: Position in sequence
            iterations_used: Number of iterations used
            task_type: Type of task
        """
        # Update token-specific history
        token_key = str(token_id)
        if token_key not in self.token_iteration_history:
            self.token_iteration_history[token_key] = {
                "count": 0,
                "total_iterations": 0,
                "avg_iterations": 0
            }
            
        token_data = self.token_iteration_history[token_key]
        token_data["count"] += 1
        token_data["total_iterations"] += iterations_used
        token_data["avg_iterations"] = token_data["total_iterations"] / token_data["count"]
        
        # Update position-specific history
        position_key = position // 10  # Group by tens
        if position_key not in self.token_positions_history:
            self.token_positions_history[position_key] = {
                "count": 0,
                "total_iterations": 0,
                "avg_iterations": 0
            }
            
        position_data = self.token_positions_history[position_key]
        position_data["count"] += 1
        position_data["total_iterations"] += iterations_used
        position_data["avg_iterations"] = position_data["total_iterations"] / position_data["count"]
        
        # Update overall statistics
        self.iteration_stats["total_tokens"] += 1
        self.iteration_stats["total_iterations"] += iterations_used
        
        # Update task-specific statistics
        if task_type not in self.iteration_stats["iterations_by_task"]:
            self.iteration_stats["iterations_by_task"][task_type] = {
                "count": 0,
                "total_iterations": 0,
                "avg_iterations": 0
            }
            
        task_data = self.iteration_stats["iterations_by_task"][task_type]
        task_data["count"] += 1
        task_data["total_iterations"] += iterations_used
        task_data["avg_iterations"] = task_data["total_iterations"] / task_data["count"]
        
        # Update position-based statistics
        position_group = position // 10 * 10  # Group by tens for stats
        if position_group not in self.iteration_stats["iterations_by_position"]:
            self.iteration_stats["iterations_by_position"][position_group] = {
                "count": 0,
                "total_iterations": 0,
                "avg_iterations": 0
            }
            
        pos_data = self.iteration_stats["iterations_by_position"][position_group]
        pos_data["count"] += 1
        pos_data["total_iterations"] += iterations_used
        pos_data["avg_iterations"] = pos_data["total_iterations"] / pos_data["count"]
    
    def get_task_strategy(self, task_type: str) -> Dict:
        """
        Get comprehensive strategy for handling a specific task type.
        
        Args:
            task_type: Type of task
            
        Returns:
            Dictionary with strategy parameters
        """
        min_iter, max_iter, threshold = self.get_iteration_params(task_type=task_type)
        
        # Build strategy dict with sensible defaults for each task type
        if task_type == "math":
            return {
                "min_iterations": min_iter,
                "max_iterations": max_iter,
                "convergence_threshold": threshold,
                "value_target": 0.85,
                "early_stopping_enabled": False,
                "backtracking_enabled": True,
                "pattern_weight": 0.8
            }
        elif task_type == "reasoning":
            return {
                "min_iterations": min_iter,
                "max_iterations": max_iter,
                "convergence_threshold": threshold,
                "value_target": 0.8,
                "early_stopping_enabled": False,
                "backtracking_enabled": True,
                "pattern_weight": 0.7
            }
        elif task_type == "creative":
            return {
                "min_iterations": min_iter,
                "max_iterations": max_iter,
                "convergence_threshold": threshold,
                "value_target": 0.7,
                "early_stopping_enabled": True,
                "backtracking_enabled": False,
                "pattern_weight": 0.5
            }
        else:  # simple, retrieval, or unknown
            return {
                "min_iterations": min_iter,
                "max_iterations": max_iter,
                "convergence_threshold": threshold,
                "value_target": 0.75,
                "early_stopping_enabled": True,
                "backtracking_enabled": False,
                "pattern_weight": 0.6
            }
    
    def get_stats(self) -> Dict:
        """Get current statistics about iteration usage"""
        avg_iters_per_token = 0
        if self.iteration_stats["total_tokens"] > 0:
            avg_iters_per_token = (self.iteration_stats["total_iterations"] / 
                                  self.iteration_stats["total_tokens"])
            
        return {
            "avg_iterations_per_token": avg_iters_per_token,
            "total_tokens_processed": self.iteration_stats["total_tokens"],
            "total_iterations": self.iteration_stats["total_iterations"],
            "task_stats": self.iteration_stats["iterations_by_task"],
            "position_stats": self.iteration_stats["iterations_by_position"]
        }