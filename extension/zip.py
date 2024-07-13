import os
import zipfile


files_to_include = ['extension/gui.py', 'extension/README.md']

# Determine the platform-specific executable files
# if os.name == 'nt':  # Windows
executable_files = ['extension/dist/GAD']
# elif os.name == 'posix':  # macOS
    # executable_files = ['age_gender_detector.app']


with zipfile.ZipFile('src/static/GAD.zip', 'w') as zipf:
    # Add the files 
    for file in files_to_include:
        zipf.write(file)

    # Add the platform-specific executable files
    for file in executable_files:
        zipf.write(file)

print('GAD.zip created successfully.')
