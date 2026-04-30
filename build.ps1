$ErrorActionPreference = 'Stop'

Write-Host "Installing dependencies from requirements.txt..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Write-Host "Building wheel package..."
python -m pip install --upgrade setuptools wheel
python -m pip install -e .

Write-Host "Build complete. Run the dashboard with: droneai or streamlit run app.py"