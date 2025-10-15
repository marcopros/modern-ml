#!/bin/bash

# 🚀 Setup Script for GitHub + GPU Server
# This script automates the setup process

set -e  # Exit on error

echo "=========================================="
echo "🚀 HYDRO FORECASTING - SETUP SCRIPT"
echo "=========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# Check if running on Mac or Linux
if [[ "$OSTYPE" == "darwin"* ]]; then
    IS_MAC=true
    print_info "Detected macOS"
else
    IS_MAC=false
    print_info "Detected Linux"
fi

# =====================================
# PART 1: GITHUB SETUP (LOCAL)
# =====================================

if [ "$1" == "init-git" ]; then
    echo ""
    echo "📦 INITIALIZING GIT REPOSITORY..."
    echo ""
    
    # Check if git is initialized
    if [ -d .git ]; then
        print_info "Git already initialized"
    else
        git init
        print_success "Git initialized"
    fi
    
    # Create .gitignore
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
dist/
*.egg-info/

# Jupyter
.ipynb_checkpoints

# Data files
*.csv
!data/prediction_mapping.csv
!data/sample_submission.csv

# Models
*.model
*.pkl
*.h5
*.pt
*.pth

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/

# Outputs
submission*.csv
!sample_submission.csv
EOF
    print_success "Created .gitignore"
    
    # Initial commit
    git add .
    git commit -m "Initial commit: Hydro forecasting project" || print_info "Nothing to commit or already committed"
    print_success "Initial commit done"
    
    echo ""
    print_info "Next steps:"
    echo "  1. Create GitHub repo: gh repo create hydro-forecasting-kaggle --private --source=. --push"
    echo "  2. Or manually: https://github.com/new"
    echo ""
fi

# =====================================
# PART 2: GPU SERVER SETUP
# =====================================

if [ "$1" == "setup-gpu" ]; then
    echo ""
    echo "🖥️  SETTING UP GPU SERVER..."
    echo ""
    
    # Check if nvidia-smi exists
    if command -v nvidia-smi &> /dev/null; then
        print_success "GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    else
        print_error "No GPU detected! Are you on the right server?"
        exit 1
    fi
    
    # Check if conda exists
    if command -v conda &> /dev/null; then
        print_success "Conda found"
    else
        print_info "Conda not found. Installing Miniconda..."
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        bash Miniconda3-latest-Linux-x86_64.sh -b
        ~/miniconda3/bin/conda init bash
        print_success "Conda installed. Please restart terminal and run this script again."
        exit 0
    fi
    
    # Create conda environment
    print_info "Creating conda environment 'hydro'..."
    conda create -n hydro python=3.10 -y || print_info "Environment already exists"
    
    print_success "Environment created"
    
    echo ""
    print_info "Next steps:"
    echo "  1. Activate environment: conda activate hydro"
    echo "  2. Install dependencies: bash setup.sh install-deps"
    echo ""
fi

# =====================================
# PART 3: INSTALL DEPENDENCIES
# =====================================

if [ "$1" == "install-deps" ]; then
    echo ""
    echo "📦 INSTALLING DEPENDENCIES..."
    echo ""
    
    # Check if in conda environment
    if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" == "base" ]; then
        print_error "Please activate conda environment first: conda activate hydro"
        exit 1
    fi
    
    print_info "Installing PyTorch with GPU support..."
    
    # Detect CUDA version
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1,2)
        print_info "Detected CUDA version: $CUDA_VERSION"
        
        if [[ "$CUDA_VERSION" == "11.8" ]]; then
            conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
        elif [[ "$CUDA_VERSION" == "12."* ]]; then
            conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
        else
            print_info "Installing PyTorch with default CUDA support"
            conda install pytorch torchvision torchaudio -c pytorch -y
        fi
    else
        print_info "Installing PyTorch CPU version (no GPU detected)"
        conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
    fi
    
    print_success "PyTorch installed"
    
    # Install other dependencies
    print_info "Installing ML libraries..."
    pip install -q pandas numpy scikit-learn matplotlib seaborn
    pip install -q lightgbm xgboost catboost
    pip install -q optuna optuna-dashboard
    pip install -q pytorch-forecasting pytorch-lightning pytorch-tabnet
    pip install -q jupyter ipykernel ipywidgets
    
    print_success "All dependencies installed"
    
    # Register Jupyter kernel
    print_info "Registering Jupyter kernel..."
    python -m ipykernel install --user --name=hydro --display-name="Python (Hydro GPU)"
    print_success "Jupyter kernel registered"
    
    # Verify GPU
    echo ""
    print_info "Verifying GPU access in PyTorch..."
    python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
    
    echo ""
    print_success "Setup complete!"
    echo ""
    print_info "You can now:"
    echo "  - Open notebooks in VS Code"
    echo "  - Or start Jupyter: jupyter lab --no-browser --port=8888"
    echo ""
fi

# =====================================
# PART 4: SYNC DATA
# =====================================

if [ "$1" == "sync-data" ]; then
    echo ""
    echo "📂 SYNCING DATA FROM LOCAL TO SERVER..."
    echo ""
    
    if [ -z "$2" ]; then
        print_error "Please provide server hostname: bash setup.sh sync-data gpu-server"
        exit 1
    fi
    
    SERVER=$2
    LOCAL_DATA_DIR="data"
    REMOTE_DATA_DIR="~/hydro-forecasting-kaggle/data"
    
    print_info "Syncing from local $LOCAL_DATA_DIR to $SERVER:$REMOTE_DATA_DIR"
    
    rsync -avz --progress \
        --exclude '*.pyc' \
        --exclude '__pycache__' \
        "$LOCAL_DATA_DIR/" \
        "$SERVER:$REMOTE_DATA_DIR/"
    
    print_success "Data synced!"
fi

# =====================================
# PART 5: DOWNLOAD RESULTS
# =====================================

if [ "$1" == "download-results" ]; then
    echo ""
    echo "⬇️  DOWNLOADING RESULTS FROM SERVER..."
    echo ""
    
    if [ -z "$2" ]; then
        print_error "Please provide server hostname: bash setup.sh download-results gpu-server"
        exit 1
    fi
    
    SERVER=$2
    
    # Create submissions folder if doesn't exist
    mkdir -p submissions
    
    print_info "Downloading submission files from $SERVER..."
    
    rsync -avz --progress \
        "$SERVER:~/hydro-forecasting-kaggle/submission*.csv" \
        submissions/
    
    print_success "Results downloaded to ./submissions/"
    
    echo ""
    print_info "Available submissions:"
    ls -lh submissions/submission*.csv
fi

# =====================================
# PART 6: MONITOR GPU
# =====================================

if [ "$1" == "monitor-gpu" ]; then
    echo ""
    echo "📊 MONITORING GPU USAGE..."
    echo ""
    
    if command -v gpustat &> /dev/null; then
        watch -n 1 gpustat -cpu
    elif command -v nvidia-smi &> /dev/null; then
        watch -n 1 nvidia-smi
    else
        print_error "No GPU monitoring tools found"
        exit 1
    fi
fi

# =====================================
# HELP
# =====================================

if [ "$1" == "help" ] || [ -z "$1" ]; then
    echo ""
    echo "USAGE: bash setup.sh <command> [options]"
    echo ""
    echo "COMMANDS:"
    echo ""
    echo "  LOCAL (run on your Mac):"
    echo "    init-git              Initialize git repository and create .gitignore"
    echo "    sync-data <server>    Sync data folder to GPU server"
    echo "    download-results <server>  Download submission files from server"
    echo ""
    echo "  REMOTE (run on GPU server):"
    echo "    setup-gpu             Check GPU and setup conda"
    echo "    install-deps          Install all Python dependencies"
    echo "    monitor-gpu           Monitor GPU usage"
    echo ""
    echo "  GENERAL:"
    echo "    help                  Show this help message"
    echo ""
    echo "EXAMPLES:"
    echo ""
    echo "  # On local Mac"
    echo "  bash setup.sh init-git"
    echo "  bash setup.sh sync-data gpu-server"
    echo ""
    echo "  # On GPU server"
    echo "  bash setup.sh setup-gpu"
    echo "  conda activate hydro"
    echo "  bash setup.sh install-deps"
    echo ""
    echo "  # Download results"
    echo "  bash setup.sh download-results gpu-server"
    echo ""
fi
