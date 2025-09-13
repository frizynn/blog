#!/bin/bash

# Script para desplegar el blog a GitHub Pages
# Autor: Juan Lebrero
# Uso: ./deploy.sh

set -e  # Salir si hay algún error

echo "🚀 Iniciando despliegue del blog..."

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para imprimir mensajes con color
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Verificar que estamos en el directorio correcto
if [ ! -f "hugo.toml" ]; then
    print_error "No se encontró hugo.toml. Asegúrate de estar en el directorio del blog."
    exit 1
fi

# Verificar que Hugo está instalado
if ! command -v hugo &> /dev/null; then
    print_error "Hugo no está instalado. Instálalo primero."
    exit 1
fi

# Verificar que git está instalado
if ! command -v git &> /dev/null; then
    print_error "Git no está instalado. Instálalo primero."
    exit 1
fi

print_status "Generando sitio estático con Hugo..."
hugo --minify

if [ $? -eq 0 ]; then
    print_success "Sitio generado correctamente"
else
    print_error "Error al generar el sitio"
    exit 1
fi

# Verificar que el submodule existe
if [ ! -d "frizynn.github.io" ]; then
    print_error "No se encontró el submodule frizynn.github.io"
    exit 1
fi

print_status "Cambiando al directorio del submodule..."
cd frizynn.github.io

# Verificar que estamos en un repositorio git (puede ser un submodule con archivo .git)
if [ ! -d ".git" ] && [ ! -f ".git" ]; then
    print_error "No se encontró un repositorio git en frizynn.github.io"
    exit 1
fi

print_status "Agregando archivos al repositorio..."
git add .

# Verificar si hay cambios para commitear
if git diff --staged --quiet; then
    print_warning "No hay cambios para commitear"
else
    print_status "Creando commit..."
    git commit -m "Update blog - $(date '+%Y-%m-%d %H:%M:%S')"
    
    if [ $? -eq 0 ]; then
        print_success "Commit creado correctamente"
    else
        print_error "Error al crear el commit"
        exit 1
    fi
fi

print_status "Haciendo push a GitHub Pages..."
git push origin main

if [ $? -eq 0 ]; then
    print_success "Push realizado correctamente"
    print_success "🎉 Blog desplegado exitosamente!"
    print_status "Tu blog estará disponible en: https://frizynn.github.io"
    print_warning "Nota: Puede tomar unos minutos para que GitHub Pages actualice el sitio"
else
    print_error "Error al hacer push"
    exit 1
fi

# Volver al directorio original
cd ..

print_success "✅ Despliegue completado!"
