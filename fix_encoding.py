import os

def clean_null_bytes_and_fix_encoding(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    # Read the file content in binary mode
                    with open(filepath, 'rb') as f:
                        content = f.read()
                    
                    # Check if null bytes or BOM exist
                    if b'\x00' in content or content.startswith(b'\xff\xfe') or content.startswith(b'\xfe\xff') or content.startswith(b'\xef\xbb\xbf'):
                        print(f"Cleaning problematic bytes in: {filepath}")
                        
                        # Remove BOM markers
                        if content.startswith(b'\xff\xfe'):
                            content = content[2:]
                        elif content.startswith(b'\xfe\xff'):
                            content = content[2:]
                        elif content.startswith(b'\xef\xbb\xbf'):
                            content = content[3:]
                            
                        # Remove null bytes
                        cleaned_content = content.replace(b'\x00', b'')
                        
                        # Try to decode as UTF-8, if it fails, use latin-1
                        try:
                            decoded = cleaned_content.decode('utf-8')
                        except UnicodeDecodeError:
                            decoded = cleaned_content.decode('latin-1')
                            print(f"  - Used latin-1 fallback encoding for {filepath}")
                        
                        # Write back as UTF-8
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(decoded)
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")

# Clean all relevant directories
for directory in ['src', 'examples', 'scripts']:
    print(f"Cleaning directory: {directory}")
    clean_null_bytes_and_fix_encoding(directory)