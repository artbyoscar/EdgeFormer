#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EdgeFormer GUI Demo

This script provides a graphical user interface to showcase EdgeFormer's capabilities.
It allows users to generate text, experiment with different settings, and visualize
model performance metrics.
"""

import os
import sys
import time
import threading
import logging
import torch
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.config import EdgeFormerConfig
from src.model.edgeformer import EdgeFormer
from better_tokenization import BetterTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("gui-demo")

class EdgeFormerGUI:
    """GUI application for demonstrating EdgeFormer capabilities."""
    
    def __init__(self, root):
        """Initialize the GUI application."""
        self.root = root
        self.root.title("EdgeFormer Demo")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
        # Model and tokenizer
        self.model = None
        self.tokenizer = None
        self.generation_thread = None
        self.stop_generation = False
        
        # Set up the main application
        self.setup_ui()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready. Load a model to begin.")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def setup_ui(self):
        """Set up the user interface components."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create notebook (tabs)
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        text_gen_tab = ttk.Frame(notebook)
        model_info_tab = ttk.Frame(notebook)
        benchmark_tab = ttk.Frame(notebook)
        
        notebook.add(text_gen_tab, text="Text Generation")
        notebook.add(model_info_tab, text="Model Info")
        notebook.add(benchmark_tab, text="Benchmarks")
        
        # Set up each tab
        self.setup_text_generation_tab(text_gen_tab)
        self.setup_model_info_tab(model_info_tab)
        self.setup_benchmark_tab(benchmark_tab)
    
    def setup_text_generation_tab(self, parent):
        """Set up the text generation tab."""
        # Top control panel
        control_frame = ttk.LabelFrame(parent, text="Model Controls")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Model load controls
        load_frame = ttk.Frame(control_frame)
        load_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(load_frame, text="Model:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.model_path_var = tk.StringVar()
        model_entry = ttk.Entry(load_frame, textvariable=self.model_path_var, width=50)
        model_entry.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(load_frame, text="Browse...", command=self.browse_model).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(load_frame, text="Load Model", command=self.load_model).grid(row=0, column=3, padx=5, pady=5)
        
        # Tokenizer selection
        ttk.Label(load_frame, text="Tokenizer:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.tokenizer_var = tk.StringVar(value="better")
        tokenizer_combo = ttk.Combobox(load_frame, textvariable=self.tokenizer_var, values=["basic", "better"])
        tokenizer_combo.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Generation parameters
        param_frame = ttk.LabelFrame(parent, text="Generation Parameters")
        param_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Parameters grid
        params_grid = ttk.Frame(param_frame)
        params_grid.pack(fill=tk.X, padx=5, pady=5)
        
        # Temperature
        ttk.Label(params_grid, text="Temperature:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.temperature_var = tk.DoubleVar(value=0.7)
        temperature_slider = ttk.Scale(params_grid, from_=0.1, to=1.5, variable=self.temperature_var, length=200, orient="horizontal")
        temperature_slider.grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(params_grid, textvariable=self.temperature_var).grid(row=0, column=2, padx=5, pady=5)
        
        # Max length
        ttk.Label(params_grid, text="Max Length:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.max_length_var = tk.IntVar(value=100)
        max_length_slider = ttk.Scale(params_grid, from_=10, to=500, variable=self.max_length_var, length=200, orient="horizontal")
        max_length_slider.grid(row=1, column=1, padx=5, pady=5)
        ttk.Label(params_grid, textvariable=self.max_length_var).grid(row=1, column=2, padx=5, pady=5)
        
        # Top-k
        ttk.Label(params_grid, text="Top-k:").grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        self.top_k_var = tk.IntVar(value=50)
        top_k_slider = ttk.Scale(params_grid, from_=1, to=100, variable=self.top_k_var, length=200, orient="horizontal")
        top_k_slider.grid(row=0, column=4, padx=5, pady=5)
        ttk.Label(params_grid, textvariable=self.top_k_var).grid(row=0, column=5, padx=5, pady=5)
        
        # Top-p
        ttk.Label(params_grid, text="Top-p:").grid(row=1, column=3, padx=5, pady=5, sticky=tk.W)
        self.top_p_var = tk.DoubleVar(value=0.9)
        top_p_slider = ttk.Scale(params_grid, from_=0.1, to=1.0, variable=self.top_p_var, length=200, orient="horizontal")
        top_p_slider.grid(row=1, column=4, padx=5, pady=5)
        ttk.Label(params_grid, textvariable=self.top_p_var).grid(row=1, column=5, padx=5, pady=5)
        
        # Use KV cache offloading
        self.kv_offload_var = tk.BooleanVar(value=False)
        kv_offload_check = ttk.Checkbutton(params_grid, text="Use KV Cache Offloading", variable=self.kv_offload_var)
        kv_offload_check.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)
        
        # Text input
        input_frame = ttk.LabelFrame(parent, text="Input Text")
        input_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.input_text = scrolledtext.ScrolledText(input_frame, height=5, wrap=tk.WORD)
        self.input_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.input_text.insert(tk.END, "EdgeFormer is a custom transformer that")
        
        # Generate button frame
        generate_frame = ttk.Frame(parent)
        generate_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(generate_frame, text="Generate Text", command=self.start_generation).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(generate_frame, text="Stop Generation", command=self.stop_generation_thread).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(generate_frame, text="Clear", command=self.clear_output).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Output text
        output_frame = ttk.LabelFrame(parent, text="Generated Text")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.output_text = scrolledtext.ScrolledText(output_frame, height=15, wrap=tk.WORD)
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def setup_model_info_tab(self, parent):
        """Set up the model info tab."""
        # Info frame
        info_frame = ttk.Frame(parent)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Model info text widget
        self.model_info_text = scrolledtext.ScrolledText(info_frame, wrap=tk.WORD)
        self.model_info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.update_model_info()
    
    def setup_benchmark_tab(self, parent):
        """Set up the benchmark tab."""
        # Controls frame
        controls_frame = ttk.Frame(parent)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(controls_frame, text="Sequence Lengths:").pack(side=tk.LEFT, padx=5, pady=5)
        self.seq_lengths_var = tk.StringVar(value="64,128,256,512,1024,2048")
        seq_lengths_entry = ttk.Entry(controls_frame, textvariable=self.seq_lengths_var, width=30)
        seq_lengths_entry.pack(side=tk.LEFT, padx=5, pady=5)
        
        ttk.Button(controls_frame, text="Run Benchmark", command=self.run_benchmark).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Plot frame
        plot_frame = ttk.Frame(parent)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create matplotlib figure
        self.figure, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))
        self.figure.tight_layout(pad=3.0)
        
        # Create canvas to display the figure
        self.canvas = FigureCanvasTkAgg(self.figure, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize plots
        self.init_plots()
    
    def init_plots(self):
        """Initialize the benchmark plots."""
        # Just create empty plots with labels
        self.ax1.set_title("Inference Time")
        self.ax1.set_xlabel("Sequence Length")
        self.ax1.set_ylabel("Time (s)")
        self.ax1.grid(True)
        
        self.ax2.set_title("Throughput")
        self.ax2.set_xlabel("Sequence Length")
        self.ax2.set_ylabel("Tokens/second")
        self.ax2.grid(True)
        
        self.canvas.draw()
    
    def browse_model(self):
        """Open a file dialog to select a model file."""
        filepath = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch Models", "*.pt"), ("All Files", "*.*")]
        )
        if filepath:
            self.model_path_var.set(filepath)
    
    def load_model(self):
        """Load the EdgeFormer model with the appropriate configuration."""
        model_path = self.model_path_var.get()
    
        if not model_path:
            messagebox.showinfo("Info", "Please specify a model path.")
            return
    
        self.status_var.set("Loading model...")
        self.root.update_idletasks()
    
        try:
            # Check if the model file exists
            if os.path.exists(model_path):
                # Try to infer the configuration from the model file
                state_dict = torch.load(model_path)
            
                # Determine max_position_embeddings from position embeddings weight
                max_position_embeddings = 128  # Default
                if 'embeddings.position_embeddings.weight' in state_dict:
                    pos_emb_shape = state_dict['embeddings.position_embeddings.weight'].shape
                    max_position_embeddings = pos_emb_shape[0]
                    self.status_var.set(f"Detected position embeddings size: {max_position_embeddings}")
                    self.root.update_idletasks()
            
                # Create model configuration with the detected position embedding size
                config = EdgeFormerConfig(
                    vocab_size=30522,
                    hidden_size=256,
                    num_hidden_layers=6,
                    num_attention_heads=8,
                    latent_size_factor=8,
                    max_position_embeddings=max_position_embeddings
                )
            
                # Initialize model with the custom configuration
                self.model = EdgeFormer(config)
            
                # Load model weights
                if os.path.isdir(model_path):
                    # Look for .pt files in directory
                    model_files = [f for f in os.listdir(model_path) if f.endswith('.pt')]
                    if model_files:
                        model_path = os.path.join(model_path, model_files[0])
            
                self.model.load_state_dict(state_dict)
                self.status_var.set(f"Model loaded from {model_path}")
            else:
                # Just use default configuration if model doesn't exist
                config = EdgeFormerConfig(
                    vocab_size=30522,
                    hidden_size=256,
                    num_hidden_layers=6,
                    num_attention_heads=8,
                    latent_size_factor=8
                )
                self.model = EdgeFormer(config)
                self.status_var.set("Using randomly initialized model")
        
            # Set model to evaluation mode
            self.model.eval()
        
            # Create tokenizer based on selection
            tokenizer_type = self.tokenizer_var.get()
            if tokenizer_type == "better":
                self.tokenizer = BetterTokenizer()
            else:
                from text_generation_demo import SimpleTokenizer
                self.tokenizer = SimpleTokenizer()
        
            # Update model info
            self.update_model_info()
        
            messagebox.showinfo("Success", "Model loaded successfully!")
    
        except Exception as e:
            error_message = str(e)
            self.status_var.set(f"Error: Loading model failed")
        
            # Check for size mismatch errors and provide a more helpful message
            if "size mismatch" in error_message:
                suggestion = "Try using the model_load_fix.py script to detect the correct configuration."
                messagebox.showerror("Size Mismatch Error", 
                                f"The model's architecture doesn't match the configuration.\n\n{error_message}\n\n{suggestion}")
            else:
                messagebox.showerror("Error", f"Failed to load model: {error_message}")
    
    def update_model_info(self):
        """Update the model info text."""
        self.model_info_text.delete(1.0, tk.END)
        
        if self.model is None:
            self.model_info_text.insert(tk.END, "No model loaded.")
            return
        
        # Get model configuration
        config = self.model.config
        
        # Calculate total parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        
        # Build info text
        info = "EdgeFormer Model Information\n"
        info += "===========================\n\n"
        info += f"Model Type: EdgeFormer\n"
        info += f"Total Parameters: {total_params:,}\n\n"
        info += "Configuration:\n"
        info += f"  - Vocabulary Size: {config.vocab_size}\n"
        info += f"  - Hidden Size: {config.hidden_size}\n"
        info += f"  - Number of Layers: {config.num_hidden_layers}\n"
        info += f"  - Number of Attention Heads: {config.num_attention_heads}\n"
        info += f"  - Latent Size Factor: {config.latent_size_factor}\n"
        info += f"  - Intermediate Size: {getattr(config, 'intermediate_size', 'N/A')}\n"
        info += f"  - Maximum Position Embeddings: {getattr(config, 'max_position_embeddings', 'N/A')}\n\n"
        info += "Tokenizer:\n"
        info += f"  - Type: {type(self.tokenizer).__name__ if self.tokenizer else 'None'}\n"
        
        if hasattr(self.tokenizer, 'vocab_size'):
            info += f"  - Vocabulary Size: {self.tokenizer.vocab_size}\n"
        
        self.model_info_text.insert(tk.END, info)
    
    def start_generation(self):
        """Start text generation in a separate thread."""
        if self.model is None:
            messagebox.showinfo("Info", "Please load a model first.")
            return
        
        prompt = self.input_text.get(1.0, tk.END).strip()
        if not prompt:
            messagebox.showinfo("Info", "Please enter a prompt.")
            return
        
        # Clear stop flag
        self.stop_generation = False
        
        # Start generation in a separate thread
        self.generation_thread = threading.Thread(
            target=self.generate_text,
            args=(prompt,)
        )
        self.generation_thread.daemon = True
        self.generation_thread.start()
    
    def stop_generation_thread(self):
        """Stop the text generation thread."""
        if self.generation_thread is not None and self.generation_thread.is_alive():
            self.stop_generation = True
    
    def clear_output(self):
        """Clear the output text."""
        self.output_text.delete(1.0, tk.END)
    
    def generate_text(self, prompt):
        """Generate text using the EdgeFormer model."""
        try:
            # Set parameters
            temperature = self.temperature_var.get()
            max_length = self.max_length_var.get()
            top_k = self.top_k_var.get()
            top_p = self.top_p_var.get()
            use_kv_offload = self.kv_offload_var.get()
            
            # Update status
            self.status_var.set("Generating text...")
            
            # Tokenize prompt
            input_ids = self.tokenizer.encode(prompt)
            input_ids = torch.tensor([input_ids], dtype=torch.long)
            
            # Create attention mask
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
            
            # Store the original prompt length
            prompt_length = input_ids.shape[1]
            
            # Set up KV cache offloading if enabled
            offload_directory = None
            if use_kv_offload:
                offload_directory = "./kv_cache_offload"
                os.makedirs(offload_directory, exist_ok=True)
            
            # Initial pass to get logits and past key values
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True
            )
            
            past_key_values = outputs["past_key_values"]
            next_token_logits = outputs["logits"][:, -1, :]
            
            # Show the prompt in the output
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, prompt)
            
            # Generate tokens one by one
            generated_ids = input_ids.clone()
            generated_text = prompt
            
            for i in range(max_length):
                if self.stop_generation:
                    self.status_var.set("Generation stopped.")
                    break
                
                # Apply temperature
                scaled_logits = next_token_logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = scaled_logits < torch.topk(scaled_logits, top_k)[0][..., -1, None]
                    scaled_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    scaled_logits[0, indices_to_remove] = float('-inf')
                
                # Sample from the filtered distribution
                probs = torch.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                # Use the helper method for continuation with KV cache
                continuation_outputs = self.model.continue_generation(next_token, past_key_values)
                past_key_values = continuation_outputs["past_key_values"]
                next_token_logits = continuation_outputs["logits"][:, -1, :]
                
                # Decode the token and update display
                next_token_text = self.tokenizer.decode([next_token.item()])
                generated_text += next_token_text
                
                # Update the UI with the new token
                self.output_text.delete(1.0, tk.END)
                self.output_text.insert(tk.END, generated_text)
                self.output_text.see(tk.END)
                self.root.update_idletasks()
                
                # Early stopping check (102 as sample EOS token)
                if next_token.item() == 102:
                    break
            
            # Clean up KV cache offload directory if it was created
            if use_kv_offload and offload_directory and os.path.exists(offload_directory):
                for file in os.listdir(offload_directory):
                    os.remove(os.path.join(offload_directory, file))
                os.rmdir(offload_directory)
            
            self.status_var.set("Text generation complete.")
        
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Failed to generate text: {str(e)}")
    
    def run_benchmark(self):
        """Run benchmark with different sequence lengths."""
        if self.model is None:
            messagebox.showinfo("Info", "Please load a model first.")
            return
        
        # Parse sequence lengths
        try:
            seq_lengths = [int(x.strip()) for x in self.seq_lengths_var.get().split(",")]
        except ValueError:
            messagebox.showerror("Error", "Invalid sequence lengths. Use comma-separated integers.")
            return
        
        self.status_var.set("Running benchmark...")
        self.root.update_idletasks()
        
        try:
            # Collect results
            times = []
            throughputs = []
            
            for seq_length in seq_lengths:
                # Create random input
                input_ids = torch.randint(0, 1000, (1, seq_length))
                attention_mask = torch.ones(1, seq_length)
                
                # Time the forward pass
                start_time = time.time()
                _ = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                elapsed = time.time() - start_time
                
                # Calculate throughput
                throughput = seq_length / elapsed
                
                times.append(elapsed)
                throughputs.append(throughput)
                
                self.status_var.set(f"Benchmarked sequence length {seq_length}...")
                self.root.update_idletasks()
            
            # Update plots
            self.ax1.clear()
            self.ax2.clear()
            
            self.ax1.set_title("Inference Time")
            self.ax1.set_xlabel("Sequence Length")
            self.ax1.set_ylabel("Time (s)")
            self.ax1.plot(seq_lengths, times, 'o-')
            self.ax1.grid(True)
            
            self.ax2.set_title("Throughput")
            self.ax2.set_xlabel("Sequence Length")
            self.ax2.set_ylabel("Tokens/second")
            self.ax2.plot(seq_lengths, throughputs, 'o-')
            self.ax2.grid(True)
            
            self.figure.tight_layout()
            self.canvas.draw()
            
            # Show summary
            summary = "Benchmark Summary:\n\n"
            for i, seq_length in enumerate(seq_lengths):
                summary += f"Sequence length {seq_length}: {times[i]:.4f}s, {throughputs[i]:.1f} tokens/s\n"
            
            messagebox.showinfo("Benchmark Results", summary)
            
            self.status_var.set("Benchmark complete.")
        
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Benchmark failed: {str(e)}")


def main():
    """Main function to start the GUI application."""
    root = tk.Tk()
    app = EdgeFormerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()