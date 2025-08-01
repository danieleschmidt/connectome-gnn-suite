#!/bin/bash
# Docker Compose development utilities for Connectome-GNN-Suite

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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
Usage: $0 COMMAND [OPTIONS]

Docker Compose utilities for Connectome-GNN-Suite development

COMMANDS:
    dev                     Start development environment
    prod                    Start production environment  
    gpu                     Start GPU-enabled training environment
    test                    Run test suite
    benchmark               Run performance benchmarks
    security                Run security scans
    docs                    Build and serve documentation
    monitoring              Start monitoring stack (Prometheus + Grafana)
    clean                   Clean up containers and volumes
    logs                    Show logs for specific service
    shell                   Open shell in development container
    jupyter                 Start Jupyter Lab
    reset                   Reset all data and restart services

OPTIONS:
    -d, --detach           Run in detached mode
    -f, --force            Force recreate containers
    --build                Build images before starting
    --pull                 Pull latest images before starting
    -h, --help             Show this help message

EXAMPLES:
    # Start development environment
    $0 dev

    # Start development with rebuild
    $0 dev --build

    # Run tests
    $0 test

    # Start GPU training environment
    $0 gpu

    # View logs for development service
    $0 logs connectome-dev

    # Open shell in development container
    $0 shell

    # Clean up everything
    $0 clean --force

SERVICE PORTS:
    8888  - Jupyter Lab (development)
    8050  - Dash applications
    8000  - Development server
    6006  - TensorBoard
    8080  - Production API
    8081  - Documentation server
    3000  - Grafana dashboard
    9090  - Prometheus metrics
    5000  - MLflow tracking
    5432  - PostgreSQL database
EOF
}

# Check if docker-compose is available
check_docker_compose() {
    if command -v docker-compose >/dev/null 2>&1; then
        COMPOSE_CMD="docker-compose"
    elif docker compose version >/dev/null 2>&1; then
        COMPOSE_CMD="docker compose"
    else
        print_error "Docker Compose not found. Please install Docker Compose."
        exit 1
    fi
}

# Ensure we're in the repository root
check_repository_root() {
    if [[ ! -f "docker-compose.yml" ]] || [[ ! -f "pyproject.toml" ]]; then
        print_error "Please run this script from the repository root directory"
        exit 1
    fi
}

# Start development environment
start_dev() {
    local args=("$@")
    print_status "Starting development environment..."
    
    # Build base arguments
    local compose_args=("up" "connectome-dev")
    
    # Add optional arguments
    for arg in "${args[@]}"; do
        case $arg in
            -d|--detach)
                compose_args+=("-d")
                ;;
            -f|--force)
                compose_args+=("--force-recreate")
                ;;
            --build)
                compose_args+=("--build")
                ;;
            --pull)
                compose_args+=("--pull")
                ;;
        esac
    done
    
    $COMPOSE_CMD "${compose_args[@]}"
    
    if [[ " ${args[*]} " =~ " -d " ]] || [[ " ${args[*]} " =~ " --detach " ]]; then
        print_success "Development environment started in detached mode"
        print_status "Jupyter Lab will be available at: http://localhost:8888"
        print_status "To view logs: $0 logs connectome-dev"
        print_status "To open shell: $0 shell"
    fi
}

# Start production environment
start_prod() {
    local args=("$@")
    print_status "Starting production environment..."
    
    local compose_args=("up" "connectome-prod")
    
    for arg in "${args[@]}"; do
        case $arg in
            -d|--detach)
                compose_args+=("-d")
                ;;
            -f|--force)
                compose_args+=("--force-recreate")
                ;;
            --build)
                compose_args+=("--build")
                ;;
        esac
    done
    
    $COMPOSE_CMD "${compose_args[@]}"
    print_success "Production environment started"
}

# Start GPU environment
start_gpu() {
    local args=("$@")
    
    # Check if NVIDIA Docker runtime is available
    if ! docker info | grep -q "nvidia"; then
        print_warning "NVIDIA Docker runtime not detected. GPU functionality may not work."
        print_warning "Install nvidia-docker2 for GPU support."
    fi
    
    print_status "Starting GPU-enabled training environment..."
    
    local compose_args=("up" "connectome-gpu")
    
    for arg in "${args[@]}"; do
        case $arg in
            -d|--detach)
                compose_args+=("-d")
                ;;
            -f|--force)
                compose_args+=("--force-recreate")
                ;;
            --build)
                compose_args+=("--build")
                ;;
        esac
    done
    
    $COMPOSE_CMD "${compose_args[@]}"
    print_success "GPU environment started"
}

# Run tests
run_tests() {
    print_status "Running test suite..."
    $COMPOSE_CMD run --rm connectome-test
    print_success "Tests completed"
}

# Run benchmarks
run_benchmarks() {
    print_status "Running performance benchmarks..."
    $COMPOSE_CMD run --rm connectome-benchmark
    print_success "Benchmarks completed"
}

# Run security scans
run_security() {
    print_status "Running security scans..."
    $COMPOSE_CMD run --rm connectome-security
    print_success "Security scans completed"
}

# Build and serve documentation
start_docs() {
    print_status "Starting documentation server..."
    $COMPOSE_CMD up -d connectome-docs
    print_success "Documentation server started at: http://localhost:8081"
}

# Start monitoring stack
start_monitoring() {
    print_status "Starting monitoring stack..."
    $COMPOSE_CMD up -d prometheus grafana
    print_success "Monitoring stack started"
    print_status "Grafana dashboard: http://localhost:3000 (admin/admin)"
    print_status "Prometheus metrics: http://localhost:9090"
}

# Clean up containers and volumes
clean_up() {
    local force=false
    
    for arg in "$@"; do
        if [[ "$arg" == "-f" ]] || [[ "$arg" == "--force" ]]; then
            force=true
        fi
    done
    
    print_status "Cleaning up containers and volumes..."
    
    if [[ "$force" == true ]]; then
        print_warning "Force cleanup: removing all containers, networks, and volumes"
        $COMPOSE_CMD down -v --remove-orphans
        docker system prune -a -f --volumes
    else
        $COMPOSE_CMD down --remove-orphans
    fi
    
    print_success "Cleanup completed"
}

# Show logs for a specific service
show_logs() {
    local service=${1:-connectome-dev}
    local follow=${2:-false}
    
    if [[ "$follow" == "true" ]] || [[ "$2" == "-f" ]] || [[ "$2" == "--follow" ]]; then
        $COMPOSE_CMD logs -f "$service"
    else
        $COMPOSE_CMD logs "$service"
    fi
}

# Open shell in development container
open_shell() {
    local service=${1:-connectome-dev}
    
    print_status "Opening shell in $service container..."
    
    # Check if container is running
    if ! $COMPOSE_CMD ps "$service" | grep -q "Up"; then
        print_status "Starting $service container..."
        $COMPOSE_CMD up -d "$service"
        sleep 5  # Wait for container to be ready
    fi
    
    $COMPOSE_CMD exec "$service" bash
}

# Start Jupyter Lab specifically
start_jupyter() {
    print_status "Starting Jupyter Lab..."
    
    if ! $COMPOSE_CMD ps connectome-dev | grep -q "Up"; then
        print_status "Starting development container..."
        $COMPOSE_CMD up -d connectome-dev
        sleep 10  # Wait for Jupyter to start
    fi
    
    print_success "Jupyter Lab is running at: http://localhost:8888"
    print_status "To view logs: $0 logs connectome-dev"
}

# Reset all data and restart services
reset_all() {
    print_warning "This will remove all data and restart services. Are you sure? (y/N)"
    read -r response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        print_status "Resetting all services..."
        $COMPOSE_CMD down -v --remove-orphans
        docker volume prune -f
        $COMPOSE_CMD up -d connectome-dev
        print_success "Services reset and restarted"
    else
        print_status "Reset cancelled"
    fi
}

# Main script logic
main() {
    check_docker_compose
    check_repository_root
    
    if [[ $# -eq 0 ]]; then
        show_help
        exit 0
    fi
    
    local command=$1
    shift
    
    case $command in
        dev)
            start_dev "$@"
            ;;
        prod)
            start_prod "$@"
            ;;
        gpu)
            start_gpu "$@"
            ;;
        test)
            run_tests
            ;;
        benchmark)
            run_benchmarks
            ;;
        security)
            run_security
            ;;
        docs)
            start_docs
            ;;
        monitoring)
            start_monitoring
            ;;
        clean)
            clean_up "$@"
            ;;
        logs)
            show_logs "$@"
            ;;
        shell)
            open_shell "$@"
            ;;
        jupyter)
            start_jupyter
            ;;
        reset)
            reset_all
            ;;
        -h|--help)
            show_help
            ;;
        *)
            print_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"