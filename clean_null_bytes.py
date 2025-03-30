import os

def clean_null_bytes(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    # Read the file content
                    with open(filepath, 'rb') as f:
                        content = f.read()
                    
                    # Check if null bytes exist
                    if b'\x00' in content:
                        print(f"Cleaning null bytes in: {filepath}")
                        # Remove null bytes
                        cleaned_content = content.replace(b'\x00', b'')
                        
                        # Write back the cleaned content
                        with open(filepath, 'wb') as f:
                            f.write(cleaned_content)
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")

# Directory to clean (use '.' for current directory)
clean_null_bytes('src')
clean_null_bytes('examples')
clean_null_bytes('scripts')