import argparse
import os
import requests
import zipfile
import io

def main(args):
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Downloading {args.dataset} dataset...")
    
    if args.dataset.lower() == "wikitext":
        # Download WikiText dataset
        url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"
        response = requests.get(url)
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(args.output_dir)
        
        print(f"Dataset downloaded and extracted to {args.output_dir}")
    else:
        print(f"Dataset {args.dataset} not supported yet.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare training datasets")
    parser.add_argument("--dataset", type=str, required=True, 
                        help="Dataset to download (e.g., wikitext)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the dataset")
    
    args = parser.parse_args()
    main(args)