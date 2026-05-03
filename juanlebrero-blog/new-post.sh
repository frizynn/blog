#!/bin/bash

# Script para crear nuevos posts con estructura de carpetas
# Autor: Juan Lebrero
# Uso: ./new-post.sh "titulo-del-post"

set -e  # Salir si hay algún error

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

# Verificar que se proporcionó un título
if [ $# -eq 0 ]; then
    print_error "Debes proporcionar un título para el post"
    echo "Uso: ./new-post.sh \"titulo-del-post\""
    exit 1
fi

# Obtener el título del post
POST_TITLE="$1"
POST_SLUG=$(echo "$POST_TITLE" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g' | sed 's/--*/-/g' | sed 's/^-\|-$//g')

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

print_status "Creando nuevo post: '$POST_TITLE'"
print_status "Slug generado: '$POST_SLUG'"

# Crear directorio para el post
POST_DIR="content/en/post/$POST_SLUG"
if [ -d "$POST_DIR" ]; then
    print_error "El directorio '$POST_DIR' ya existe"
    exit 1
fi

mkdir -p "$POST_DIR"

# Obtener la fecha actual
CURRENT_DATE=$(date '+%Y-%m-%dT%H:%M:%S-03:00')

# Crear el archivo index.md con frontmatter
cat > "$POST_DIR/index.md" << EOF
---
date: $CURRENT_DATE
draft: false
title: '$POST_TITLE'
type: post
---

# $POST_TITLE

Escribe tu contenido aquí...

## Introducción

## Desarrollo

## Conclusión
EOF

print_success "Post creado en: $POST_DIR/index.md"

# Crear directorio para recursos del post (imágenes, etc.)
mkdir -p "$POST_DIR/images"

print_success "Directorio de imágenes creado en: $POST_DIR/images"

# Mostrar instrucciones
echo ""
print_status "📝 Próximos pasos:"
echo "1. Edita el archivo: $POST_DIR/index.md"
echo "2. Agrega imágenes en: $POST_DIR/images/"
echo "3. Ejecuta './deploy.sh' para publicar"
echo ""
print_status "💡 Comandos útiles:"
echo "• Ver preview: hugo server"
echo "• Publicar: ./deploy.sh"
echo "• Editar: code $POST_DIR/index.md"

print_success "✅ Post '$POST_TITLE' creado exitosamente!"
