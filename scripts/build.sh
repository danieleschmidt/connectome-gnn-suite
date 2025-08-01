#!/bin/bash
# Build script for Connectome-GNN-Suite Docker images
# Supports multiple build targets and platforms

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
BUILD_TARGET="production"
PYTHON_VERSION="3.11"
PYTORCH_VERSION="2.0.0"
CUDA_VERSION="11.8"
BUILD_PLATFORM="linux/amd64"
PUSH_IMAGE=false
TAG_LATEST=false
REGISTRY=""
NO_CACHE=false

# Function to print colored output
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

# Function to show usage
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Build Docker images for Connectome-GNN-Suite

OPTIONS:
    -t, --target TARGET         Build target (production, development, testing, security, gpu-production)
                               Default: production
    -p, --python VERSION        Python version (3.9, 3.10, 3.11)
                               Default: 3.11
    --pytorch VERSION          PyTorch version
                               Default: 2.0.0
    --cuda VERSION             CUDA version for GPU builds
                               Default: 11.8
    --platform PLATFORM        Target platform (linux/amd64, linux/arm64)
                               Default: linux/amd64
    --push                     Push image to registry after build
    --latest                   Also tag as 'latest'
    --registry REGISTRY        Docker registry URL
    --no-cache                 Build without using cache
    -h, --help                 Show this help message

EXAMPLES:
    # Build production image
    $0 --target production

    # Build development image with Python 3.11
    $0 --target development --python 3.11

    # Build GPU-enabled image
    $0 --target gpu-production --cuda 11.8

    # Build and push to registry
    $0 --target production --push --registry gcr.io/my-project

    # Build for ARM64 (Apple Silicon)
    $0 --target development --platform linux/arm64

AVAILABLE TARGETS:
    production      - Optimized production image
    development     - Full development environment with tools
    testing         - Testing environment with test dependencies
    security        - Security scanning and validation
    gpu-production  - GPU-enabled production image with CUDA support
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--target)
            BUILD_TARGET="$2"
            shift 2
            ;;
        -p|--python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --pytorch)
            PYTORCH_VERSION="$2"
            shift 2
            ;;
        --cuda)
            CUDA_VERSION="$2"
            shift 2
            ;;
        --platform)
            BUILD_PLATFORM="$2"
            shift 2
            ;;
        --push)
            PUSH_IMAGE=true
            shift
            ;;
        --latest)
            TAG_LATEST=true
            shift
            ;;
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate build target
case $BUILD_TARGET in
    production|development|testing|security|gpu-production)
        ;;
    *)
        print_error "Invalid build target: $BUILD_TARGET"
        print_error "Valid targets: production, development, testing, security, gpu-production"
        exit 1
        ;;
esac

# Set image name
IMAGE_NAME="connectome-gnn-suite"
if [[ -n "$REGISTRY" ]]; then
    FULL_IMAGE_NAME="$REGISTRY/$IMAGE_NAME"
else
    FULL_IMAGE_NAME="$IMAGE_NAME"
fi

# Create tag based on target and version
TAG="${BUILD_TARGET}-py${PYTHON_VERSION}"
if [[ "$BUILD_TARGET" == "gpu-production" ]]; then
    TAG="${TAG}-cuda${CUDA_VERSION}"
fi

FULL_TAG="$FULL_IMAGE_NAME:$TAG"

print_status "Building Connectome-GNN-Suite Docker image"
print_status "Target: $BUILD_TARGET"
print_status "Python version: $PYTHON_VERSION"
print_status "PyTorch version: $PYTORCH_VERSION"
if [[ "$BUILD_TARGET" == "gpu-production" ]]; then
    print_status "CUDA version: $CUDA_VERSION"
fi
print_status "Platform: $BUILD_PLATFORM"
print_status "Image tag: $FULL_TAG"

# Build Docker build arguments
BUILD_ARGS=(
    "--build-arg" "PYTHON_VERSION=$PYTHON_VERSION"
    "--build-arg" "PYTORCH_VERSION=$PYTORCH_VERSION"
    "--build-arg" "BUILD_ENV=$BUILD_TARGET"
    "--target" "$BUILD_TARGET"
    "--platform" "$BUILD_PLATFORM"
    "--tag" "$FULL_TAG"
)

if [[ "$BUILD_TARGET" == "gpu-production" ]]; then
    BUILD_ARGS+=("--build-arg" "CUDA_VERSION=$CUDA_VERSION")
fi

if [[ "$NO_CACHE" == true ]]; then
    BUILD_ARGS+=("--no-cache")
fi

# Add latest tag if requested
if [[ "$TAG_LATEST" == true ]]; then
    BUILD_ARGS+=("--tag" "$FULL_IMAGE_NAME:latest")
fi

# Ensure we're in the repository root
if [[ ! -f "pyproject.toml" ]] || [[ ! -f "Dockerfile" ]]; then
    print_error "Please run this script from the repository root directory"
    exit 1
fi

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    print_error "Docker is not running or not accessible"
    exit 1
fi

# Check if buildx is available for multi-platform builds
if [[ "$BUILD_PLATFORM" != "linux/amd64" ]]; then
    if ! docker buildx version >/dev/null 2>&1; then
        print_error "Docker buildx is required for multi-platform builds"
        exit 1
    fi
    
    # Use buildx for multi-platform builds
    print_status "Using Docker buildx for multi-platform build"
    BUILD_CMD="docker buildx build"
    BUILD_ARGS+=("--load")  # Load the image into local Docker daemon
else
    BUILD_CMD="docker build"
fi

# Build the image
print_status "Starting Docker build..."
echo "Command: $BUILD_CMD ${BUILD_ARGS[*]} ."

if $BUILD_CMD "${BUILD_ARGS[@]}" .; then
    print_success "Docker image built successfully: $FULL_TAG"
else
    print_error "Docker build failed"
    exit 1
fi

# Push image if requested
if [[ "$PUSH_IMAGE" == true ]]; then
    if [[ -z "$REGISTRY" ]]; then
        print_warning "No registry specified, pushing to default registry"
    fi
    
    print_status "Pushing image to registry..."
    if docker push "$FULL_TAG"; then
        print_success "Image pushed successfully: $FULL_TAG"
    else
        print_error "Failed to push image"
        exit 1
    fi
    
    # Push latest tag if it was created
    if [[ "$TAG_LATEST" == true ]]; then
        if docker push "$FULL_IMAGE_NAME:latest"; then
            print_success "Latest tag pushed successfully: $FULL_IMAGE_NAME:latest"
        else
            print_error "Failed to push latest tag"
            exit 1
        fi
    fi
fi

# Display final information
print_success "Build completed successfully!"
echo
print_status "Image details:"
echo "  Image: $FULL_TAG"
echo "  Target: $BUILD_TARGET"
echo "  Python: $PYTHON_VERSION"
echo "  PyTorch: $PYTORCH_VERSION"
if [[ "$BUILD_TARGET" == "gpu-production" ]]; then
    echo "  CUDA: $CUDA_VERSION"
fi
echo "  Platform: $BUILD_PLATFORM"
echo
print_status "To run the image:"
case $BUILD_TARGET in
    development)
        echo "  docker run -it --rm -p 8888:8888 $FULL_TAG"
        ;;
    production)
        echo "  docker run --rm $FULL_TAG"
        ;;
    gpu-production)
        echo "  docker run --gpus all --rm $FULL_TAG"
        ;;
    testing)
        echo "  docker run --rm $FULL_TAG"
        ;;
    security)
        echo "  docker run --rm $FULL_TAG"
        ;;
esac

echo
print_status "For more options, see: docker run --help"