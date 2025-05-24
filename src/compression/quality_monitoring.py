#!/usr/bin/env python3
"""
Real-Time Quality Monitoring System for EdgeFormer
Continuous monitoring and adaptive adjustment of compression quality
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import json
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.compression.int4_quantization import INT4Quantizer

@dataclass
class QualityMetrics:
    """Quality metrics for compression monitoring"""
    timestamp: float
    layer_name: str
    compression_ratio: float
    accuracy_loss: float
    inference_latency_ms: float
    memory_usage_mb: float
    output_variance: float
    gradient_norm: float
    attention_entropy: float = 0.0
    feature_preservation: float = 0.0

@dataclass
class QualityThresholds:
    """Quality thresholds for monitoring"""
    max_accuracy_loss: float = 1.0  # Maximum acceptable accuracy loss %
    max_latency_ms: float = 100.0   # Maximum acceptable latency
    min_compression: float = 6.0    # Minimum acceptable compression ratio
    max_memory_mb: float = 512.0    # Maximum memory usage
    min_feature_preservation: float = 0.95  # Minimum feature preservation
    attention_entropy_threshold: float = 0.1  # Attention quality threshold

class QualityMonitor:
    """
    Real-time quality monitoring system for EdgeFormer compression
    """
    
    def __init__(self, history_size: int = 1000):
        self.int4_quantizer = INT4Quantizer()
        self.history_size = history_size
        self.quality_history: deque = deque(maxlen=history_size)
        self.alerts: List[str] = []
        self.thresholds = QualityThresholds()
        
        # Monitoring state
        self.monitoring_active = False
        self.adaptive_mode = True
        self.alert_callbacks: List[Callable] = []
        
    def register_alert_callback(self, callback: Callable[[str, QualityMetrics], None]):
        """Register callback for quality alerts"""
        self.alert_callbacks.append(callback)
    
    def start_monitoring(self, model: nn.Module, thresholds: Optional[QualityThresholds] = None):
        """Start quality monitoring for a model"""
        if thresholds:
            self.thresholds = thresholds
        
        self.monitoring_active = True
        self.quality_history.clear()
        self.alerts.clear()
        
        print(f"🔍 Quality monitoring started")
        print(f"   Max accuracy loss: {self.thresholds.max_accuracy_loss}%")
        print(f"   Max latency: {self.thresholds.max_latency_ms}ms")
        print(f"   Min compression: {self.thresholds.min_compression}x")
    
    def monitor_layer_compression(self, layer_name: str, original_tensor: torch.Tensor,
                                compressed_tensor: torch.Tensor, metadata: Dict) -> QualityMetrics:
        """Monitor quality metrics for a single layer compression"""
        
        start_time = time.time()
        
        # Calculate basic metrics
        compression_ratio = metadata.get("compression", 1.0)
        
        # Dequantize for accuracy measurement
        if "scale" in metadata and "zero_point" in metadata:
            dequantized = self.int4_quantizer.dequantize_tensor(
                compressed_tensor, metadata["scale"], metadata["zero_point"]
            )
        else:
            dequantized = compressed_tensor.float()
        
        # Accuracy loss
        accuracy_loss = torch.mean(torch.abs(original_tensor - dequantized)).item() * 100
        
        # Memory usage estimation
        original_size = original_tensor.numel() * 4 / (1024 * 1024)  # MB
        compressed_size = original_size / compression_ratio
        
        # Output variance (measure of information preservation)
        output_variance = torch.var(dequantized).item()
        
        # Gradient norm (if available)
        gradient_norm = 0.0
        if original_tensor.grad is not None:
            gradient_norm = torch.norm(original_tensor.grad).item()
        
        # Feature preservation score
        feature_preservation = self._calculate_feature_preservation(original_tensor, dequantized)
        
        # Attention entropy (for attention layers)
        attention_entropy = 0.0
        if "attention" in layer_name.lower():
            attention_entropy = self._calculate_attention_entropy(dequantized)
        
        # Create quality metrics
        metrics = QualityMetrics(
            timestamp=time.time(),
            layer_name=layer_name,
            compression_ratio=compression_ratio,
            accuracy_loss=accuracy_loss,
            inference_latency_ms=(time.time() - start_time) * 1000,
            memory_usage_mb=compressed_size,
            output_variance=output_variance,
            gradient_norm=gradient_norm,
            attention_entropy=attention_entropy,
            feature_preservation=feature_preservation
        )
        
        # Store in history
        if self.monitoring_active:
            self.quality_history.append(metrics)
            
            # Check thresholds and generate alerts
            self._check_quality_thresholds(metrics)
        
        return metrics
    
    def _calculate_feature_preservation(self, original: torch.Tensor, compressed: torch.Tensor) -> float:
        """Calculate feature preservation score"""
        try:
            # Normalize tensors
            orig_norm = torch.nn.functional.normalize(original.flatten(), dim=0)
            comp_norm = torch.nn.functional.normalize(compressed.flatten(), dim=0)
            
            # Cosine similarity as preservation metric
            similarity = torch.dot(orig_norm, comp_norm).item()
            return max(0.0, similarity)  # Ensure non-negative
        except:
            return 0.5  # Default fallback
    
    def _calculate_attention_entropy(self, attention_tensor: torch.Tensor) -> float:
        """Calculate attention entropy for attention layers"""
        try:
            # Flatten and normalize to probability distribution
            flattened = attention_tensor.flatten()
            probs = torch.softmax(flattened, dim=0)
            
            # Calculate entropy
            entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
            return entropy / np.log(len(flattened))  # Normalize by max possible entropy
        except:
            return 0.0
    
    def _check_quality_thresholds(self, metrics: QualityMetrics):
        """Check if quality metrics exceed thresholds"""
        alerts = []
        
        if metrics.accuracy_loss > self.thresholds.max_accuracy_loss:
            alerts.append(f"HIGH_ACCURACY_LOSS: {metrics.accuracy_loss:.3f}% > {self.thresholds.max_accuracy_loss}%")
        
        if metrics.inference_latency_ms > self.thresholds.max_latency_ms:
            alerts.append(f"HIGH_LATENCY: {metrics.inference_latency_ms:.2f}ms > {self.thresholds.max_latency_ms}ms")
        
        if metrics.compression_ratio < self.thresholds.min_compression:
            alerts.append(f"LOW_COMPRESSION: {metrics.compression_ratio:.1f}x < {self.thresholds.min_compression}x")
        
        if metrics.memory_usage_mb > self.thresholds.max_memory_mb:
            alerts.append(f"HIGH_MEMORY: {metrics.memory_usage_mb:.1f}MB > {self.thresholds.max_memory_mb}MB")
        
        if metrics.feature_preservation < self.thresholds.min_feature_preservation:
            alerts.append(f"LOW_FEATURE_PRESERVATION: {metrics.feature_preservation:.3f} < {self.thresholds.min_feature_preservation}")
        
        if "attention" in metrics.layer_name.lower() and metrics.attention_entropy < self.thresholds.attention_entropy_threshold:
            alerts.append(f"LOW_ATTENTION_ENTROPY: {metrics.attention_entropy:.3f} < {self.thresholds.attention_entropy_threshold}")
        
        # Store and notify alerts
        for alert in alerts:
            self.alerts.append(f"[{time.strftime('%H:%M:%S')}] {metrics.layer_name}: {alert}")
            
            # Call registered callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert, metrics)
                except Exception as e:
                    print(f"⚠️  Alert callback failed: {e}")
    
    def get_quality_summary(self) -> Dict:
        """Get comprehensive quality summary"""
        if not self.quality_history:
            return {"status": "no_data"}
        
        metrics_list = list(self.quality_history)
        
        # Calculate statistics
        accuracy_losses = [m.accuracy_loss for m in metrics_list]
        compression_ratios = [m.compression_ratio for m in metrics_list]
        latencies = [m.inference_latency_ms for m in metrics_list]
        feature_preservations = [m.feature_preservation for m in metrics_list]
        
        summary = {
            "monitoring_period": {
                "start_time": metrics_list[0].timestamp,
                "end_time": metrics_list[-1].timestamp,
                "total_layers_monitored": len(metrics_list)
            },
            "accuracy_statistics": {
                "mean_accuracy_loss": np.mean(accuracy_losses),
                "max_accuracy_loss": np.max(accuracy_losses),
                "std_accuracy_loss": np.std(accuracy_losses),
                "threshold_violations": len([x for x in accuracy_losses if x > self.thresholds.max_accuracy_loss])
            },
            "compression_statistics": {
                "mean_compression": np.mean(compression_ratios),
                "min_compression": np.min(compression_ratios),
                "std_compression": np.std(compression_ratios),
                "threshold_violations": len([x for x in compression_ratios if x < self.thresholds.min_compression])
            },
            "performance_statistics": {
                "mean_latency_ms": np.mean(latencies),
                "max_latency_ms": np.max(latencies),
                "std_latency_ms": np.std(latencies),
                "threshold_violations": len([x for x in latencies if x > self.thresholds.max_latency_ms])
            },
            "quality_statistics": {
                "mean_feature_preservation": np.mean(feature_preservations),
                "min_feature_preservation": np.min(feature_preservations),
                "std_feature_preservation": np.std(feature_preservations),
                "threshold_violations": len([x for x in feature_preservations if x < self.thresholds.min_feature_preservation])
            },
            "alerts": {
                "total_alerts": len(self.alerts),
                "recent_alerts": self.alerts[-10:] if self.alerts else [],
                "alert_rate": len(self.alerts) / len(metrics_list) if metrics_list else 0
            },
            "overall_health": self._calculate_overall_health()
        }
        
        return summary
    
    def _calculate_overall_health(self) -> Dict:
        """Calculate overall system health score"""
        if not self.quality_history:
            return {"score": 0.0, "status": "no_data"}
        
        metrics_list = list(self.quality_history)
        
        # Health components (0-1 scale, 1 = perfect)
        accuracy_health = 1.0 - (np.mean([m.accuracy_loss for m in metrics_list]) / self.thresholds.max_accuracy_loss)
        compression_health = np.mean([min(m.compression_ratio / self.thresholds.min_compression, 1.0) for m in metrics_list])
        latency_health = 1.0 - (np.mean([m.inference_latency_ms for m in metrics_list]) / self.thresholds.max_latency_ms)
        feature_health = np.mean([m.feature_preservation for m in metrics_list])
        
        # Clip to 0-1 range
        accuracy_health = max(0.0, min(1.0, accuracy_health))
        compression_health = max(0.0, min(1.0, compression_health))
        latency_health = max(0.0, min(1.0, latency_health))
        feature_health = max(0.0, min(1.0, feature_health))
        
        # Weighted overall score
        overall_score = (
            accuracy_health * 0.3 +
            compression_health * 0.25 +
            latency_health * 0.25 +
            feature_health * 0.2
        )
        
        # Determine status
        if overall_score >= 0.9:
            status = "excellent"
        elif overall_score >= 0.8:
            status = "good"
        elif overall_score >= 0.6:
            status = "fair"
        elif overall_score >= 0.4:
            status = "poor"
        else:
            status = "critical"
        
        return {
            "score": overall_score,
            "status": status,
            "components": {
                "accuracy_health": accuracy_health,
                "compression_health": compression_health,
                "latency_health": latency_health,
                "feature_health": feature_health
            }
        }
    
    def generate_quality_report(self, save_path: Optional[str] = None) -> Dict:
        """Generate comprehensive quality report"""
        
        report = {
            "report_timestamp": time.time(),
            "report_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "monitoring_configuration": {
                "thresholds": {
                    "max_accuracy_loss": self.thresholds.max_accuracy_loss,
                    "max_latency_ms": self.thresholds.max_latency_ms,
                    "min_compression": self.thresholds.min_compression,
                    "max_memory_mb": self.thresholds.max_memory_mb,
                    "min_feature_preservation": self.thresholds.min_feature_preservation
                },
                "history_size": self.history_size,
                "adaptive_mode": self.adaptive_mode
            },
            "quality_summary": self.get_quality_summary(),
            "detailed_metrics": [
                {
                    "timestamp": m.timestamp,
                    "layer_name": m.layer_name,
                    "compression_ratio": m.compression_ratio,
                    "accuracy_loss": m.accuracy_loss,
                    "inference_latency_ms": m.inference_latency_ms,
                    "feature_preservation": m.feature_preservation
                }
                for m in list(self.quality_history)[-100:]  # Last 100 entries
            ]
        }
        
        if save_path:
            output_path = Path(save_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"📄 Quality report saved to: {output_path}")
        
        return report
    
    def print_real_time_dashboard(self):
        """Print real-time monitoring dashboard"""
        if not self.quality_history:
            print("📊 No monitoring data available")
            return
        
        summary = self.get_quality_summary()
        health = summary["overall_health"]
        
        print(f"\n{'='*60}")
        print("📊 EDGEFORMER QUALITY MONITORING DASHBOARD")
        print(f"{'='*60}")
        
        # Overall health
        status_emoji = {
            "excellent": "🟢",
            "good": "🟡", 
            "fair": "🟠",
            "poor": "🔴",
            "critical": "🆘"
        }
        
        print(f"\n🏥 OVERALL HEALTH: {status_emoji.get(health['status'], '❓')} {health['status'].upper()}")
        print(f"   Health Score: {health['score']:.1%}")
        
        # Key metrics
        print(f"\n📈 KEY METRICS:")
        print(f"   Avg Accuracy Loss: {summary['accuracy_statistics']['mean_accuracy_loss']:.3f}%")
        print(f"   Avg Compression: {summary['compression_statistics']['mean_compression']:.1f}x")
        print(f"   Avg Latency: {summary['performance_statistics']['mean_latency_ms']:.2f}ms")
        print(f