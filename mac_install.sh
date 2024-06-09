#!/bin/bash

# Create a virtual environment if not already created
if [ ! -d "ocrtools" ]; then
    python3 -m venv ocrtools
fi

# Activate the virtual environment
source ocrtools/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install the package
pip install -e .

# Notify user
echo "Installation complete. You can now run the Streamlit app using 'streamlit run app.py'"