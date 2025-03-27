# scripts/preprocess_wikitext.py
import os
import argparse

def preprocess_wikitext(input_dir, output_dir):
    # Process train, valid, and test files
    for split in ["train", "valid", "test"]:
        input_path = os.path.join(input_dir, f"wiki.{split}.tokens")
        output_path = os.path.join(output_dir, f"{split}.txt")
        
        if os.path.exists(input_path):
            with open(input_path, 'r', encoding='utf-8') as infile, \
                 open(output_path, 'w', encoding='utf-8') as outfile:
                
                # Process line by line
                for line in infile:
                    # Skip empty lines and section headers
                    if line.strip() and not line.startswith(" = "):
                        outfile.write(line)
            
            print(f"Preprocessed {input_path} -> {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess WikiText dataset")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing the raw WikiText files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the preprocessed files")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    preprocess_wikitext(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()