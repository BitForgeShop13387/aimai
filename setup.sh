#!/bin/bash

# Kreiraj .env fajl
echo "API_KEY=\"your_key_here\"" > config/.env

# Dodaj .env u .gitignore
echo ".env" >> .gitignore

# Instaliraj zavisnosti
pip install -r requirements.txt

echo "✅ Setup završen! Ne zaboravi da izmeniš .env fajl sa svojim ključevima."
