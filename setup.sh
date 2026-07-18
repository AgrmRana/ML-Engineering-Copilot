#!/bin/bash

echo "Setting up ML Workspace AI..."

# Create data directories
mkdir -p backend/data/uploads
mkdir -p backend/data/chroma

# Backend setup
echo "Setting up backend..."
cd backend
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
echo "Backend setup complete. Please edit backend/.env with your configuration."

# Frontend setup
echo "Setting up frontend..."
cd ../frontend
npm install
echo "Frontend setup complete."

echo "Setup complete!"
echo "To start the application:"
echo "1. Edit backend/.env with your OpenAI API key and database configuration"
echo "2. Start PostgreSQL (or use docker-compose up -d)"
echo "3. Start backend: cd backend && source venv/bin/activate && uvicorn api.main:app --reload"
echo "4. Start frontend: cd frontend && npm run dev"
