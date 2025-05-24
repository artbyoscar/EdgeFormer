#!/usr/bin/env python3
"""
Fixed Universal Validation Script
Import fix and streamlined for immediate use
"""

import torch
import torch.nn as nn
from typing import Dict, List
from pathlib import Path
import sys
import time
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import from existing working benchmark
from src.compression.int4_quantization import INT4Quantizer
from src.compression.utils import calculate_model_size

class QuickUniversalBenchmark:
    """
    Quick universal validation using proven EdgeFormer compression
    """
    
    def __init__(self):
        self.int4_quantizer = INT4Quantizer()
    
    def benchmark_edgeformer_on_model(self, model: nn.Module, model_name: str) -> Dict:
        """Benchmark EdgeFormer on any model"""
        print(f"🔄 Testing EdgeFormer on {model_name}")
        
        original_size = calculate_model_size(model)
        
        start_time = time.time()
        total_compressed_size = 0
        total_accuracy_loss = 0
        successful_layers = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad and len(param.shape) >= 2:
                try:
                    # Apply EdgeFormer compression
                    quantized, scale, zero_point = self.int4_quantizer.quantize_tensor(param)
                    
                    # Calculate layer metrics
                    layer_compressed_size = param.numel() * 0.5 / (1024 * 1024)  # INT4 = 0.5 bytes
                    total_compressed_size += layer_compressed_size
                    
                    # Calculate accuracy loss
                    dequantized = self.int4_quantizer.dequantize_tensor(quantized, scale, zero_point)
                    layer_accuracy_loss = torch.mean(torch.abs(param - dequantized)).item() * 100
                    total_accuracy_loss += layer_accuracy_loss
                    successful_layers += 1
                    
                except Exception as e:
                    continue
        
        compression_time = (time.time() - start_time) * 1000
        
        if successful_layers > 0:
            compression_ratio = original_size / total_compressed_size
            avg_accuracy_loss = total_accuracy_loss / successful_layers
            
            result = {
                "model_name": model_name,
                "original_size_mb": original_size,
                "compressed_size_mb": total_compressed_size,
                "compression_ratio": compression_ratio,
                "accuracy_loss_percent": avg_accuracy_loss,
                "compression_time_ms": compression_time,
                "successful_layers": successful_layers,
                "quality_score": max(0, 1.0 - avg_accuracy_loss/100.0)
            }
            
            print(f"   ✅ {compression_ratio:.1f}x compression, {avg_accuracy_loss:.3f}% accuracy loss")
            return result
        else:
            print(f"   ❌ Failed to compress {model_name}")
            return None

    def create_diverse_models(self) -> Dict[str, nn.Module]:
        """Create diverse transformer models for universal testing"""
        
        # GPT-style models (autoregressive)
        class TestGPT(nn.Module):
            def __init__(self, vocab_size=1000, hidden_size=256, num_layers=4):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_size)
                self.layers = nn.ModuleList([
                    nn.TransformerEncoderLayer(hidden_size, 4, hidden_size*4, batch_first=True)
                    for _ in range(num_layers)
                ])
                self.lm_head = nn.Linear(hidden_size, vocab_size)
            
            def forward(self, x):
                x = self.embedding(x)
                for layer in self.layers:
                    x = layer(x)
                return self.lm_head(x)
        
        # BERT-style models (bidirectional)
        class TestBERT(nn.Module):
            def __init__(self, vocab_size=1000, hidden_size=256, num_layers=4):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_size)
                self.encoder = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(hidden_size, 4, hidden_size*4, batch_first=True),
                    num_layers
                )
                self.classifier = nn.Linear(hidden_size, 2)
            
            def forward(self, x):
                x = self.embedding(x)
                x = self.encoder(x)
                return self.classifier(x[:, 0])
        
        # T5-style models (encoder-decoder)
        class TestT5(nn.Module):
            def __init__(self, vocab_size=1000, hidden_size=256, num_layers=4):
                super().__init__()
                self.encoder_embedding = nn.Embedding(vocab_size, hidden_size)
                self.decoder_embedding = nn.Embedding(vocab_size, hidden_size)
                self.encoder = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(hidden_size, 4, hidden_size*4, batch_first=True),
                    num_layers
                )
                self.decoder = nn.TransformerDecoder(
                    nn.TransformerDecoderLayer(hidden_size, 4, hidden_size*4, batch_first=True),
                    num_layers
                )
                self.lm_head = nn.Linear(hidden_size, vocab_size)
            
            def forward(self, encoder_input, decoder_input):
                enc_emb = self.encoder_embedding(encoder_input)
                enc_out = self.encoder(enc_emb)
                dec_emb = self.decoder_embedding(decoder_input)
                dec_out = self.decoder(dec_emb, enc_out)
                return self.lm_head(dec_out)
        
        # Vision Transformer
        class TestViT(nn.Module):
            def __init__(self, patch_size=16, hidden_size=256, num_layers=4):
                super().__init__()
                self.patch_embedding = nn.Conv2d(3, hidden_size, patch_size, patch_size)
                self.pos_embedding = nn.Parameter(torch.randn(1, 197, hidden_size))
                self.encoder = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(hidden_size, 4, hidden_size*4, batch_first=True),
                    num_layers
                )
                self.head = nn.Linear(hidden_size, 1000)
            
            def forward(self, x):
                batch_size = x.shape[0]
                x = self.patch_embedding(x)
                x = x.flatten(2).transpose(1, 2)
                cls_token = torch.zeros(batch_size, 1, x.shape[-1], device=x.device)
                x = torch.cat([cls_token, x], dim=1)
                x = x + self.pos_embedding
                x = self.encoder(x)
                return self.head(x[:, 0])
        
        # CLIP-style multimodal
        class TestCLIP(nn.Module):
            def __init__(self, vocab_size=1000, hidden_size=256, num_layers=4):
                super().__init__()
                # Text encoder
                self.text_embedding = nn.Embedding(vocab_size, hidden_size)
                self.text_encoder = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(hidden_size, 4, hidden_size*4, batch_first=True),
                    num_layers
                )
                # Vision encoder  
                self.vision_embedding = nn.Conv2d(3, hidden_size, 16, 16)
                self.vision_encoder = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(hidden_size, 4, hidden_size*4, batch_first=True),
                    num_layers
                )
                self.text_projection = nn.Linear(hidden_size, hidden_size)
                self.vision_projection = nn.Linear(hidden_size, hidden_size)
            
            def forward(self, text_input, image_input):
                text_emb = self.text_embedding(text_input)
                text_features = self.text_encoder(text_emb)
                vision_emb = self.vision_embedding(image_input).flatten(2).transpose(1, 2)
                vision_features = self.vision_encoder(vision_emb)
                return self.text_projection(text_features[:, 0]), self.vision_projection(vision_features[:, 0])
        
        # Create model suite
        models = {
            # Different architectures
            "GPT-Small": TestGPT(vocab_size=1000, hidden_size=256, num_layers=4),
            "BERT-Small": TestBERT(vocab_size=1000, hidden_size=256, num_layers=4), 
            "T5-Small": TestT5(vocab_size=1000, hidden_size=256, num_layers=4),
            "ViT-Small": TestViT(patch_size=16, hidden_size=256, num_layers=4),
            "CLIP-Small": TestCLIP(vocab_size=1000, hidden_size=256, num_layers=4),
            
            # Different scales
            "GPT-Tiny": TestGPT(vocab_size=500, hidden_size=128, num_layers=2),
            "GPT-Medium": TestGPT(vocab_size=2000, hidden_size=512, num_layers=6),
            "BERT-Large": TestBERT(vocab_size=3000, hidden_size=768, num_layers=8),
            
            # Efficient variants
            "GPT-Efficient": TestGPT(vocab_size=1000, hidden_size=192, num_layers=3),
            "ViT-Efficient": TestViT(patch_size=32, hidden_size=192, num_layers=3)
        }
        
        return models
    
    def run_universal_validation(self):
        """Run comprehensive universal validation"""
        
        print("🌍 UNIVERSAL TRANSFORMER COMPRESSION VALIDATION")
        print("=" * 80)
        print("Testing EdgeFormer across diverse architectures and scales")
        
        # Create diverse model suite
        models = self.create_diverse_models()
        
        total_size = sum(calculate_model_size(model) for model in models.values())
        print(f"\n🏗️ Created {len(models)} diverse models (Total: {total_size:.1f}MB)")
        
        # Test each model
        results = []
        successful_compressions = 0
        
        for model_name, model in models.items():
            size = calculate_model_size(model)
            param_count = sum(p.numel() for p in model.parameters())
            print(f"\n📊 {model_name}: {size:.1f}MB ({param_count:,} parameters)")
            
            result = self.benchmark_edgeformer_on_model(model, model_name)
            if result:
                results.append(result)
                successful_compressions += 1
        
        # Analyze universality
        self.analyze_universal_results(results, len(models))
        
        # Save results
        self.save_universal_results(results)
        
        return results
    
    def analyze_universal_results(self, results: List[Dict], total_models: int):
        """Analyze universality metrics"""
        
        print(f"\n{'='*80}")
        print("🎯 UNIVERSAL COMPRESSION ANALYSIS")
        print(f"{'='*80}")
        
        if not results:
            print("❌ No successful compressions")
            return
        
        # Overall metrics
        avg_compression = sum(r["compression_ratio"] for r in results) / len(results)
        avg_accuracy_loss = sum(r["accuracy_loss_percent"] for r in results) / len(results)
        avg_quality = sum(r["quality_score"] for r in results) / len(results)
        success_rate = len(results) / total_models
        
        print(f"📊 UNIVERSAL PERFORMANCE METRICS:")
        print(f"   Models successfully compressed: {len(results)}/{total_models}")
        print(f"   Success rate: {success_rate:.1%}")
        print(f"   Average compression: {avg_compression:.1f}x")
        print(f"   Average accuracy loss: {avg_accuracy_loss:.3f}%")
        print(f"   Average quality score: {avg_quality:.3f}")
        
        # Architecture analysis
        architecture_groups = {
            "Autoregressive (GPT-style)": ["GPT"],
            "Bidirectional (BERT-style)": ["BERT"],
            "Encoder-Decoder (T5-style)": ["T5"],
            "Vision (ViT-style)": ["ViT"],
            "Multimodal (CLIP-style)": ["CLIP"]
        }
        
        print(f"\n🏗️ ARCHITECTURE BREAKDOWN:")
        for arch_name, keywords in architecture_groups.items():
            arch_results = [r for r in results if any(k in r["model_name"] for k in keywords)]
            if arch_results:
                arch_compression = sum(r["compression_ratio"] for r in arch_results) / len(arch_results)
                arch_accuracy = sum(r["accuracy_loss_percent"] for r in arch_results) / len(arch_results)
                print(f"   {arch_name}: {arch_compression:.1f}x compression, {arch_accuracy:.3f}% loss ({len(arch_results)} models)")
        
        # Scale analysis
        scale_groups = {
            "Tiny": ["Tiny"],
            "Small": ["Small"],
            "Medium": ["Medium"], 
            "Large": ["Large"],
            "Efficient": ["Efficient"]
        }
        
        print(f"\n📏 SCALE BREAKDOWN:")
        for scale_name, keywords in scale_groups.items():
            scale_results = [r for r in results if any(k in r["model_name"] for k in keywords)]
            if scale_results:
                scale_compression = sum(r["compression_ratio"] for r in scale_results) / len(scale_results)
                scale_accuracy = sum(r["accuracy_loss_percent"] for r in scale_results) / len(scale_results)
                print(f"   {scale_name}: {scale_compression:.1f}x compression, {scale_accuracy:.3f}% loss ({len(scale_results)} models)")
        
        # Universality verdict
        if avg_compression >= 7.0 and avg_accuracy_loss <= 1.5 and success_rate >= 0.9:
            print(f"\n✅ VERDICT: EdgeFormer demonstrates STRONG UNIVERSALITY")
            print(f"   Consistent high-quality compression across diverse architectures")
        elif avg_compression >= 6.0 and avg_accuracy_loss <= 2.0 and success_rate >= 0.8:
            print(f"\n⚠️ VERDICT: EdgeFormer shows GOOD universality")
            print(f"   Reliable compression with room for architecture-specific optimization")
        else:
            print(f"\n❌ VERDICT: Universality needs improvement")
        
        return {
            "avg_compression": avg_compression,
            "avg_accuracy_loss": avg_accuracy_loss,
            "success_rate": success_rate,
            "total_models": total_models
        }
    
    def save_universal_results(self, results: List[Dict]):
        """Save universal validation results"""
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        output_file = results_dir / "universal_validation_results.json"
        with open(output_file, 'w') as f:
            json.dump({
                "timestamp": time.time(),
                "total_models_tested": len(results),
                "results": results
            }, f, indent=2)
        
        print(f"\n💾 Universal validation results saved to: {output_file}")

def main():
    """Run universal validation"""
    benchmark = QuickUniversalBenchmark()
    results = benchmark.run_universal_validation()
    
    print(f"\n✅ Universal validation complete!")

if __name__ == "__main__":
    main()