# PowerShell setup script for Windows

Write-Host "Setting up ML Workspace AI..."

# Create data directories
New-Item -ItemType Directory -Force -Path backend\data\uploads
New-Item -ItemType Directory -Force -Path backend\data\chroma

# Backend setup
Write-Host "Setting up backend..."
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
Write-Host "Backend setup complete. Please edit backend\.env with your configuration."

# Frontend setup
Write-Host "Setting up frontend..."
cd ..\frontend
npm install
Write-Host "Frontend setup complete."

Write-Host "Setup complete!"
Write-Host "To start the application:"
Write-Host "1. Edit backend\.env with your OpenAI API key and database configuration"
Write-Host "2. Start PostgreSQL (or use docker-compose up -d)"
Write-Host "3. Start backend: cd backend; .\venv\Scripts\activate; uvicorn api.main:app --reload"
Write-Host "4. Start frontend: cd frontend; npm run dev"
