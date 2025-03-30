# scripts/verify_limo_dependencies.py
import sys
import importlib
import subprocess

def check_import(module_name):
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name} is properly installed")
        return True
    except ImportError:
        print(f"❌ {module_name} is NOT installed")
        return False

def check_nltk_data(data_name):
    import nltk
    try:
        nltk.data.find(f'tokenizers/{data_name}')
        print(f"✅ NLTK {data_name} is properly installed")
        return True
    except LookupError:
        print(f"❌ NLTK {data_name} is NOT installed")
        return False

def main():
    # Check required Python packages
    required_packages = [
        'matplotlib', 'seaborn', 'pandas', 'sklearn', 
        'textstat', 'nltk', 'tqdm'
    ]
    
    missing_packages = []
    for package in required_packages:
        if not check_import(package):
            missing_packages.append(package)
    
    # Check required NLTK data
    required_nltk_data = [
        'punkt', 'punkt_tab', 'wordnet', 'stopwords', 
        'averaged_perceptron_tagger'
    ]
    
    missing_nltk = []
    for data in required_nltk_data:
        if not check_nltk_data(data):
            missing_nltk.append(data)
    
    # Install missing dependencies if any
    if missing_packages or missing_nltk:
        print("\n🔄 Missing dependencies detected. Installing them now...")
        
        if missing_packages:
            subprocess.call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
        
        if missing_nltk:
            for data in missing_nltk:
                subprocess.call([sys.executable, '-c', f'"import nltk; nltk.download(\'{data}\')"'])
        
        print("\n✅ Dependency installation complete. Please run this script again to verify.")
    else:
        print("\n✅ All LIMO training dependencies are properly installed!")

if __name__ == "__main__":
    main()