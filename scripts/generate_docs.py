import os
import argparse
import inspect
import importlib
import sys
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def generate_module_docs(module_name, output_dir):
    try:
        module = importlib.import_module(module_name)
        
        # Get all classes and functions in the module
        module_items = inspect.getmembers(module)
        classes = [item for item in module_items if inspect.isclass(item[1]) and item[1].__module__ == module.__name__]
        functions = [item for item in module_items if inspect.isfunction(item[1]) and item[1].__module__ == module.__name__]
        
        # Create the markdown content
        content = f"# {module_name}\n\n"
        
        # Module docstring
        if module.__doc__:
            content += f"{module.__doc__.strip()}\n\n"
        
        # Classes
        if classes:
            content += "## Classes\n\n"
            for name, cls in classes:
                content += f"### {name}\n\n"
                if cls.__doc__:
                    content += f"{cls.__doc__.strip()}\n\n"
                
                # Methods
                methods = inspect.getmembers(cls, predicate=inspect.isfunction)
                if methods:
                    content += "#### Methods\n\n"
                    for method_name, method in methods:
                        if not method_name.startswith('_') or method_name == '__init__':
                            content += f"##### `{method_name}`\n\n"
                            if method.__doc__:
                                content += f"{method.__doc__.strip()}\n\n"
                            
                            # Method signature
                            signature = inspect.signature(method)
                            content += f"```python\n{method_name}{signature}\n```\n\n"
        
        # Functions
        if functions:
            content += "## Functions\n\n"
            for name, func in functions:
                content += f"### {name}\n\n"
                if func.__doc__:
                    content += f"{func.__doc__.strip()}\n\n"
                
                # Function signature
                signature = inspect.signature(func)
                content += f"```python\n{name}{signature}\n```\n\n"
        
        # Write to file
        filename = os.path.join(output_dir, f"{module_name.replace('.', '_')}.md")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return filename
    
    except ImportError:
        print(f"Could not import module {module_name}")
        return None

def main(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Modules to document
    modules = [
        'src.model.edgeformer',
        'src.model.multi_head_latent_attention',
        'src.model.sparse_mlp',
        'src.model.transformer_block',
        'src.model.config',
        'src.utils.device',
        'src.utils.rdna3_optimizations',
        'src.utils.weight_quantization',
        'src.utils.training_utils',
        'src.utils.data_augmentation',
        'src.utils.kv_cache_offload',
    ]
    
    # Generate docs for each module
    generated_files = []
    for module in modules:
        filename = generate_module_docs(module, args.output_dir)
        if filename:
            generated_files.append((module, filename))
    
    # Create an index.md file
    with open(os.path.join(args.output_dir, 'index.md'), 'w', encoding='utf-8') as f:
        f.write(f"# EdgeFormer Documentation\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Modules\n\n")
        
        for module, filename in generated_files:
            rel_path = os.path.relpath(filename, args.output_dir)
            f.write(f"- [{module}]({rel_path})\n")
    
    print(f"Documentation generated in {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate documentation for EdgeFormer")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the documentation")
    
    args = parser.parse_args()
    main(args)