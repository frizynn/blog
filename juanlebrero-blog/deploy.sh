#!/bin/bash

# Script para desplegar el blog a GitHub Pages
# Autor: Juan Lebrero
# Uso: ./deploy.sh

set -euo pipefail  # Salir si hay algún error o variable indefinida

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

# Verificar que rsync está instalado
if ! command -v rsync &> /dev/null; then
    print_error "rsync no está instalado. Instálalo primero."
    exit 1
fi

BUILD_DIR=".hugo-deploy"
PUBLISH_DIR="frizynn.github.io"

print_status "Generando sitio estático con Hugo en ${BUILD_DIR}..."
hugo --minify --cleanDestinationDir --gc --destination "${BUILD_DIR}"

if [ $? -eq 0 ]; then
    print_success "Sitio generado correctamente"
else
    print_error "Error al generar el sitio"
    exit 1
fi

# Verificar que el submodule existe
if [ ! -d "${PUBLISH_DIR}" ]; then
    print_error "No se encontró el submodule ${PUBLISH_DIR}"
    exit 1
fi

if [ ! -d "${PUBLISH_DIR}/.git" ] && [ ! -f "${PUBLISH_DIR}/.git" ]; then
    print_error "No se encontró un repositorio git en ${PUBLISH_DIR}. Ejecuta: git submodule update --init --recursive"
    exit 1
fi

print_status "Sincronizando artefactos hacia ${PUBLISH_DIR} (preservando .git)..."
rsync -a --delete --exclude '.git' "${BUILD_DIR}/" "${PUBLISH_DIR}/"

print_status "Cambiando al directorio del submodule..."
cd "${PUBLISH_DIR}"

# Asegurar que el submodule esté en la rama main (no detached HEAD)
CURRENT_BRANCH="$(git symbolic-ref --short -q HEAD || true)"
if [ -z "${CURRENT_BRANCH}" ]; then
    print_warning "Submodule en detached HEAD. Cambiando a rama main..."
    git checkout -B main origin/main
fi

print_status "Sincronizando rama main del submodule..."
git fetch origin
git checkout main
git pull --ff-only origin main

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
git push origin HEAD:main

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
