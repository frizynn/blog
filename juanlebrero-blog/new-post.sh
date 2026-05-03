#!/bin/bash

# Script para crear nuevos posts con estructura de carpetas
# Autor: Juan Lebrero
# Uso:
#   ./new-post.sh "Mi nuevo post"            # Crea EN + ES
#   ./new-post.sh --lang en "Mi nuevo post"  # Solo EN
#   ./new-post.sh --lang es "Mi nuevo post"  # Solo ES

set -euo pipefail  # Salir si hay algún error, variables sin definir o fallos en pipes

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

# Parsear argumentos: --lang en|es|both (default: both) y el título posicional
LANG_OPT="both"
POST_TITLE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --lang)
            if [ $# -lt 2 ]; then
                print_error "La opción --lang requiere un valor: en, es o both"
                exit 1
            fi
            LANG_OPT="$2"
            shift 2
            ;;
        --lang=*)
            LANG_OPT="${1#--lang=}"
            shift
            ;;
        -h|--help)
            echo "Uso:"
            echo "  ./new-post.sh \"Mi nuevo post\"            # Crea EN + ES"
            echo "  ./new-post.sh --lang en \"Mi nuevo post\"  # Solo EN"
            echo "  ./new-post.sh --lang es \"Mi nuevo post\"  # Solo ES"
            exit 0
            ;;
        --)
            shift
            POST_TITLE="${1:-}"
            break
            ;;
        -*)
            print_error "Opción desconocida: $1"
            exit 1
            ;;
        *)
            if [ -z "$POST_TITLE" ]; then
                POST_TITLE="$1"
            else
                print_error "Argumento posicional inesperado: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# Validar --lang
case "$LANG_OPT" in
    en|es|both) ;;
    *)
        print_error "Valor inválido para --lang: '$LANG_OPT' (esperado: en, es o both)"
        exit 1
        ;;
esac

# Verificar que se proporcionó un título
if [ -z "$POST_TITLE" ]; then
    print_error "Debes proporcionar un título para el post"
    echo "Uso: ./new-post.sh [--lang en|es|both] \"titulo-del-post\""
    exit 1
fi

# Generar slug a partir del título (pipeline original)
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
print_status "Idiomas a generar: '$LANG_OPT'"

# Determinar lista de idiomas
if [ "$LANG_OPT" = "both" ]; then
    LANGS=(en es)
else
    LANGS=("$LANG_OPT")
fi

# Obtener la fecha actual
CURRENT_DATE=$(date '+%Y-%m-%dT%H:%M:%S-03:00')

# Crear el post para cada idioma solicitado
CREATED_DIRS=()
for lang in "${LANGS[@]}"; do
    POST_DIR="content/${lang}/post/${POST_SLUG}"
    if [ -d "$POST_DIR" ]; then
        print_error "El directorio '$POST_DIR' ya existe"
        exit 1
    fi

    mkdir -p "$POST_DIR"

    if [ "$lang" = "es" ]; then
        # Para español: mismo título por defecto + recordatorio para traducir
        cat > "$POST_DIR/index.md" << EOF
---
date: $CURRENT_DATE
draft: false
title: '$POST_TITLE'
type: post
translationKey: "$POST_SLUG"
---

<!-- TODO: Traducir este post al español. El título y el contenido están en inglés por defecto. -->

# $POST_TITLE

Escribe tu contenido aquí...

## Introducción

## Desarrollo

## Conclusión
EOF
    else
        cat > "$POST_DIR/index.md" << EOF
---
date: $CURRENT_DATE
draft: false
title: '$POST_TITLE'
type: post
translationKey: "$POST_SLUG"
---

# $POST_TITLE

Escribe tu contenido aquí...

## Introducción

## Desarrollo

## Conclusión
EOF
    fi

    print_success "Post creado en: $POST_DIR/index.md"

    # Crear directorio para recursos del post (imágenes, etc.)
    mkdir -p "$POST_DIR/images"
    print_success "Directorio de imágenes creado en: $POST_DIR/images"

    CREATED_DIRS+=("$POST_DIR")
done

# Mostrar instrucciones
echo ""
print_status "📝 Próximos pasos:"
step=1
for dir in "${CREATED_DIRS[@]}"; do
    echo "${step}. Edita el archivo: $dir/index.md"
    step=$((step + 1))
done
echo "${step}. Agrega imágenes en cada carpeta images/ correspondiente"
step=$((step + 1))
echo "${step}. Ejecuta './deploy.sh' para publicar"
echo ""
print_status "💡 Comandos útiles:"
echo "• Ver preview: hugo server"
echo "• Publicar: ./deploy.sh"
for dir in "${CREATED_DIRS[@]}"; do
    echo "• Editar: code $dir/index.md"
done

print_success "✅ Post '$POST_TITLE' creado exitosamente!"
