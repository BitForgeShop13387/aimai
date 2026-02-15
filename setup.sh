#!/bin/bash

# Kreiraj .env fajl

# Dodaj .env u .gitignore
echo ".env" >> .gitignore

# Instaliraj zavisnosti
pip install -r requirements.txt

echo "✅ Setup završen! Ne zaboravi da izmeniš .env fajl sa svojim ključevima."

echo ""
echo "🔒 Bezbednosna provera..."
if [ ! -f "config/.env" ]; then
    echo "⚠️  config/.env fajl ne postoji!"
    echo "💡 Kopiraj primer: cp config/.env.example config/.env"
    echo "🔑 Zatim dodaj svoje stvarne API ključeve u config/.env"
else
    echo "✅ config/.env postoji (lokalno - NIKAD se ne commituje)"
fi

echo ""
echo "✅ Setup završen! Projekat je spreman za razvoj."
