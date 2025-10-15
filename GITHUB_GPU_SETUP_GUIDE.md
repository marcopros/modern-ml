# 🚀 GUIDA: GitHub Repo → VS Code Remote GPU

## 📋 OVERVIEW
Questa guida ti mostrerà come:
1. Creare una GitHub repository del progetto
2. Configurare VS Code Remote SSH
3. Clonare e lavorare su macchina GPU remota

---

## PARTE 1: CREARE GITHUB REPOSITORY

### Step 1: Inizializza Git localmente

```bash
# Vai nella directory del progetto
cd "/Users/marcoprosperi/Desktop/Università/append_consulting_project 5"

# Inizializza git (se non già fatto)
git init

# Crea .gitignore
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
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb_checkpoints/

# Data files (too large for git)
*.csv
*.parquet
*.pkl
*.pickle
*.h5
*.hdf5

# Keep only small files
!data/prediction_mapping.csv
!data/sample_submission.csv

# Models (too large)
*.model
*.pkl
*.joblib
*.h5
*.pt
*.pth
*.onnx

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Outputs
submission*.csv
!sample_submission.csv
EOF

# Add all files
git add .

# First commit
git commit -m "Initial commit: Hydro raw material forecasting project

- Multiple solution notebooks (V1-V7)
- LightGBM/XGBoost/CatBoost implementations
- GPU-accelerated pipeline ready
- Feature engineering with 100+ features
- Optuna hyperparameter optimization
"
```

### Step 2: Crea Repository su GitHub

**Opzione A: Via Web (più semplice)**

1. Vai su https://github.com/new
2. Repository name: `hydro-forecasting-kaggle` (o nome a tua scelta)
3. Description: `Kaggle competition: Hydro raw material delivery forecasting with quantile regression`
4. **Private** (se vuoi mantenerlo privato)
5. **NON** inizializzare con README, .gitignore, o license (li hai già)
6. Click "Create repository"

**Opzione B: Via GitHub CLI**

```bash
# Installa GitHub CLI se non ce l'hai
brew install gh  # macOS
# oppure scarica da https://cli.github.com/

# Login
gh auth login

# Crea repo
gh repo create hydro-forecasting-kaggle --private --source=. --remote=origin --push

# Questo comando:
# - Crea il repo su GitHub
# - Aggiunge remote origin
# - Fa push automaticamente
```

### Step 3: Collega e Push (se hai usato Opzione A)

```bash
# Aggiungi remote (sostituisci YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/hydro-forecasting-kaggle.git

# Verifica
git remote -v

# Push
git branch -M main
git push -u origin main
```

---

## PARTE 2: SETUP VS CODE REMOTE SSH

### Step 1: Installa Estensioni VS Code

Apri VS Code e installa:
1. **Remote - SSH** (ms-vscode-remote.remote-ssh)
2. **Remote - SSH: Editing Configuration Files** (ms-vscode-remote.remote-ssh-edit)
3. **Remote Explorer** (ms-vscode.remote-explorer)

```bash
# Oppure via command line
code --install-extension ms-vscode-remote.remote-ssh
code --install-extension ms-vscode.remote-explorer
```

### Step 2: Configura SSH Config

```bash
# Apri il file SSH config
code ~/.ssh/config

# Aggiungi la configurazione del tuo server GPU
# Esempio:
```

```
Host gpu-server
    HostName your-gpu-server.com  # oppure IP: 123.45.67.89
    User your-username
    Port 22
    IdentityFile ~/.ssh/id_rsa  # la tua chiave SSH
    ForwardAgent yes
    ServerAliveInterval 60
    ServerAliveCountMax 5
```

**Parametri comuni:**
- **HostName**: IP o hostname del server GPU
- **User**: il tuo username sul server
- **Port**: solitamente 22, ma controlla con il tuo provider
- **IdentityFile**: percorso alla tua chiave SSH privata

### Step 3: Setup Chiave SSH (se non ce l'hai)

```bash
# Genera nuova chiave SSH
ssh-keygen -t ed25519 -C "your.email@example.com"
# Premi Enter per salvare in ~/.ssh/id_ed25519
# Opzionale: aggiungi passphrase

# Copia la chiave pubblica sul server GPU
ssh-copy-id -i ~/.ssh/id_ed25519.pub your-username@your-gpu-server.com

# Oppure manualmente:
cat ~/.ssh/id_ed25519.pub
# Copia l'output e aggiungilo a ~/.ssh/authorized_keys sul server
```

### Step 4: Testa Connessione SSH

```bash
# Test connessione
ssh gpu-server

# Se funziona, esci
exit
```

---

## PARTE 3: CONNETTI VS CODE AL SERVER GPU

### Step 1: Connetti VS Code

1. **Apri VS Code**
2. Premi `Cmd+Shift+P` (macOS) o `Ctrl+Shift+P` (Windows/Linux)
3. Cerca: `Remote-SSH: Connect to Host...`
4. Seleziona `gpu-server` (o il nome che hai dato)
5. Scegli piattaforma: **Linux**
6. Aspetta che VS Code si connetta e installi il server remoto

### Step 2: Clona Repository sul Server GPU

Una volta connesso, apri il terminale integrato in VS Code (`Ctrl+\`` o `Cmd+\``):

```bash
# Configura Git (se prima volta)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Clona il repository
cd ~
git clone https://github.com/YOUR_USERNAME/hydro-forecasting-kaggle.git

# Oppure con SSH (se hai configurato chiavi SSH su GitHub)
git clone git@github.com:YOUR_USERNAME/hydro-forecasting-kaggle.git

# Entra nella directory
cd hydro-forecasting-kaggle
```

### Step 3: Apri Folder in VS Code

1. `File` → `Open Folder...`
2. Seleziona `~/hydro-forecasting-kaggle`
3. Click **OK**

---

## PARTE 4: SETUP AMBIENTE PYTHON SU GPU

### Step 1: Verifica GPU

```bash
# Controlla GPU disponibili
nvidia-smi

# Dovresti vedere output tipo:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.0   |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |===============================+======================+======================|
# |   0  NVIDIA A100        Off  | 00000000:00:04.0 Off |                    0 |
# | N/A   30C    P0    42W / 250W |      0MiB / 40960MiB |      0%      Default |
```

### Step 2: Setup Conda/Venv

**Opzione A: Conda (raccomandato per GPU)**

```bash
# Crea environment
conda create -n hydro python=3.10 -y
conda activate hydro

# Installa PyTorch con CUDA (controlla versione CUDA con nvidia-smi)
# Per CUDA 11.8:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Per CUDA 12.1:
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Verifica GPU disponibile in PyTorch
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**Opzione B: venv**

```bash
# Crea virtual environment
python3 -m venv venv
source venv/bin/activate

# Installa PyTorch GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Installa Dependencies

```bash
# Core ML libraries
pip install pandas numpy scikit-learn matplotlib seaborn

# Gradient Boosting GPU
pip install lightgbm
pip install xgboost
pip install catboost  # Ha supporto GPU nativo!

# Hyperparameter tuning
pip install optuna optuna-dashboard

# Deep Learning for Time Series
pip install pytorch-forecasting pytorch-lightning
pip install pytorch-tabnet

# Feature engineering
pip install tsfresh featuretools

# Jupyter
pip install jupyter ipykernel ipywidgets

# Register kernel for Jupyter
python -m ipykernel install --user --name=hydro --display-name="Python (Hydro GPU)"
```

### Step 4: Crea requirements.txt

```bash
# Genera requirements
pip freeze > requirements.txt

# Oppure crea manualmente con versioni specifiche:
cat > requirements.txt << 'EOF'
# Core
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Gradient Boosting
lightgbm>=4.0.0
xgboost>=2.0.0
catboost>=1.2.0

# Deep Learning
torch>=2.0.0
torchvision>=0.15.0
pytorch-forecasting>=1.0.0
pytorch-lightning>=2.0.0
pytorch-tabnet>=4.0.0

# Optimization
optuna>=3.3.0
optuna-dashboard>=0.13.0

# Feature Engineering
tsfresh>=0.20.0
featuretools>=1.27.0

# Jupyter
jupyter>=1.0.0
ipykernel>=6.25.0
ipywidgets>=8.1.0
EOF

pip install -r requirements.txt
```

---

## PARTE 5: UPLOAD DATA FILES

### Opzione A: rsync (raccomandato per file grandi)

Da **tuo Mac locale**:

```bash
# Upload data folder
rsync -avz --progress \
  "/Users/marcoprosperi/Desktop/Università/append_consulting_project 5/data/" \
  gpu-server:~/hydro-forecasting-kaggle/data/

# Questo comando:
# -a: archive mode (preserva permessi, timestamp)
# -v: verbose
# -z: compressione
# --progress: mostra progresso
```

### Opzione B: scp

```bash
# Upload singoli file
scp "/Users/marcoprosperi/Desktop/Università/append_consulting_project 5/data/kernel/receivals.csv" \
  gpu-server:~/hydro-forecasting-kaggle/data/kernel/

# Upload intera cartella
scp -r "/Users/marcoprosperi/Desktop/Università/append_consulting_project 5/data" \
  gpu-server:~/hydro-forecasting-kaggle/
```

### Opzione C: Via VS Code

1. Apri **Explorer** in VS Code (connesso al remote)
2. Right-click sulla cartella `data/`
3. Upload files/folders dal menu

### Opzione D: Download da Kaggle direttamente sul server GPU

```bash
# Installa Kaggle CLI
pip install kaggle

# Setup Kaggle API
mkdir -p ~/.kaggle
nano ~/.kaggle/kaggle.json
# Incolla il tuo API token da https://www.kaggle.com/settings
# {"username":"your-username","key":"your-api-key"}

chmod 600 ~/.kaggle/kaggle.json

# Download dataset
kaggle competitions download -c your-competition-name
unzip your-competition-name.zip -d data/
```

---

## PARTE 6: ESEGUI NOTEBOOK SU GPU

### Step 1: Avvia Jupyter su Server GPU

**Opzione A: Jupyter Notebook in VS Code (raccomandato)**

1. Apri `solution_gpu_phase1.ipynb` in VS Code
2. Click **Select Kernel** in alto a destra
3. Scegli `Python (Hydro GPU)`
4. Esegui celle normalmente!

**Opzione B: Jupyter Lab via Port Forwarding**

Sul server GPU:
```bash
# Avvia Jupyter Lab
jupyter lab --no-browser --port=8888

# Output mostrerà:
# http://localhost:8888/lab?token=abc123...
```

Da tuo Mac locale:
```bash
# Forward porta 8888
ssh -L 8888:localhost:8888 gpu-server
```

Apri browser: `http://localhost:8888/lab?token=abc123...`

### Step 2: Verifica GPU Usage

Durante training, monitora GPU:

```bash
# Watch GPU usage ogni 1 secondo
watch -n 1 nvidia-smi

# Oppure usa gpustat (più leggibile)
pip install gpustat
gpustat -i 1
```

### Step 3: Esegui Training

```python
# In notebook, verifica GPU
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")

# CatBoost GPU
from catboost import CatBoostRegressor

model = CatBoostRegressor(
    loss_function='Quantile:alpha=0.2',
    task_type='GPU',  # ← IMPORTANTE!
    devices='0',
    iterations=10000
)
model.fit(X_train, y_train)

# XGBoost GPU
import xgboost as xgb

params = {
    'tree_method': 'gpu_hist',  # ← IMPORTANTE!
    'device': 'cuda',
    'objective': 'reg:quantileerror'
}
model = xgb.train(params, dtrain)
```

---

## PARTE 7: WORKFLOW TIPICO

### 1. Modifica codice localmente su Mac
```bash
# Fai modifiche nei notebook
# Commit
git add solution_gpu_phase1.ipynb
git commit -m "Add improved feature engineering"
git push
```

### 2. Pull su server GPU
```bash
# In VS Code Remote terminal
git pull
```

### 3. Esegui training su GPU
```bash
# Esegui notebook in VS Code
# Oppure via command line:
jupyter nbconvert --to notebook --execute solution_gpu_phase1.ipynb --output solution_gpu_phase1_executed.ipynb
```

### 4. Download risultati
```bash
# Da Mac locale
rsync -avz gpu-server:~/hydro-forecasting-kaggle/submission_gpu_*.csv ./submissions/

# Oppure usa VS Code: right-click su file → Download
```

### 5. Upload su Kaggle
```bash
# Upload submission
kaggle competitions submit -c competition-name -f submission_gpu_ensemble.csv -m "GPU ensemble model"
```

---

## 🎯 PROVIDER GPU CONSIGLIATI

### Cloud Provider con GPU

1. **Google Colab Pro/Pro+** ($10-50/mese)
   - ✅ Setup zero
   - ✅ Jupyter integrato
   - ❌ Timeout sessione
   - GPU: T4, A100

2. **Kaggle Notebooks** (GRATIS!)
   - ✅ 30h/settimana GPU gratis
   - ✅ Dataset già disponibile
   - ✅ Zero setup
   - GPU: P100, T4

3. **Lambda Labs** ($0.50-2.50/ora)
   - ✅ GPU dedicate
   - ✅ Jupyter preinstallato
   - ✅ SSH access
   - GPU: A100, H100, RTX 4090

4. **Vast.ai** ($0.20-1.50/ora)
   - ✅ Economico
   - ✅ Molte opzioni GPU
   - ❌ Più complesso da configurare
   - GPU: varie

5. **RunPod** ($0.30-2.00/ora)
   - ✅ Templates preconfigurati
   - ✅ Jupyter + VS Code integrati
   - ✅ Facile da usare
   - GPU: A100, RTX 4090, H100

6. **Google Cloud Platform** (pay-as-you-go)
   - ✅ Affidabile
   - ❌ Più costoso (~$2-3/ora)
   - GPU: A100, V100, T4

### Setup per Kaggle Notebooks (GRATIS!)

```python
# In Kaggle notebook, abilita GPU:
# Settings → Accelerator → GPU T4 x2

# Installa packages
!pip install catboost optuna pytorch-forecasting

# Clone repo (se pubblico)
!git clone https://github.com/YOUR_USERNAME/hydro-forecasting-kaggle.git
%cd hydro-forecasting-kaggle

# Esegui training!
```

---

## 📋 CHECKLIST COMPLETA

### Setup Iniziale
- [ ] Crea .gitignore
- [ ] git init & commit
- [ ] Crea repo GitHub
- [ ] git push

### VS Code Remote
- [ ] Installa estensioni Remote-SSH
- [ ] Configura ~/.ssh/config
- [ ] Setup chiave SSH
- [ ] Test connessione SSH
- [ ] Connetti VS Code a server GPU

### Server GPU Setup
- [ ] Verifica GPU con nvidia-smi
- [ ] Crea conda environment
- [ ] Installa PyTorch GPU
- [ ] Installa dependencies (requirements.txt)
- [ ] Clone repository
- [ ] Upload data files

### Esecuzione
- [ ] Apri notebook in VS Code Remote
- [ ] Select kernel Python (Hydro GPU)
- [ ] Esegui celle
- [ ] Monitor GPU con nvidia-smi
- [ ] Download submissions
- [ ] Upload su Kaggle

---

## 🆘 TROUBLESHOOTING

### "GPU not found" in PyTorch
```bash
# Controlla versione CUDA
nvidia-smi  # guarda "CUDA Version"

# Reinstalla PyTorch con CUDA corretta
# Se CUDA 11.8:
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### CatBoost non usa GPU
```python
# Verifica supporto GPU
from catboost.utils import get_gpu_device_count
print(f"GPUs detected: {get_gpu_device_count()}")

# Se 0, reinstalla:
pip uninstall catboost
pip install catboost --no-cache-dir
```

### Out of Memory GPU
```python
# Riduci batch size o num_samples
train_df_gpu = create_gpu_training_samples(
    receivals, purchase_orders, materials, 
    n_samples=10000  # invece di 50000
)

# Oppure usa gradient accumulation in PyTorch
```

### SSH Connection Timeout
```bash
# Aggiungi a ~/.ssh/config
ServerAliveInterval 60
ServerAliveCountMax 5
TCPKeepAlive yes
```

---

## 🎓 BEST PRACTICES

1. **Git branching**: Usa branch per esperimenti
   ```bash
   git checkout -b experiment/tft-model
   # fai modifiche
   git commit -am "Try TFT model"
   git push -u origin experiment/tft-model
   ```

2. **Backup modelli**: Salva checkpoints
   ```python
   model.save_model('checkpoints/model_epoch_100.cbm')
   ```

3. **Monitor costs**: Setta budget alerts sul cloud provider

4. **Use tmux/screen**: Per sessioni persistenti
   ```bash
   tmux new -s training
   # esegui training
   # Ctrl+B, D per detach
   # tmux attach -t training per riattaccare
   ```

5. **Logging**: Salva logs di training
   ```python
   import logging
   logging.basicConfig(filename='training.log', level=logging.INFO)
   ```

---

## 🚀 QUICK START RECAP

```bash
# 1. Setup locale
cd "/Users/marcoprosperi/Desktop/Università/append_consulting_project 5"
git init
gh repo create hydro-forecasting-kaggle --private --source=. --push

# 2. Configura SSH
code ~/.ssh/config  # aggiungi configurazione server GPU

# 3. Connetti VS Code
# Cmd+Shift+P → Remote-SSH: Connect to Host → gpu-server

# 4. Sul server GPU (in VS Code Remote terminal)
git clone https://github.com/YOUR_USERNAME/hydro-forecasting-kaggle.git
cd hydro-forecasting-kaggle
conda create -n hydro python=3.10 -y
conda activate hydro
pip install -r requirements.txt

# 5. Upload data
# Da Mac locale:
rsync -avz data/ gpu-server:~/hydro-forecasting-kaggle/data/

# 6. Esegui!
# Apri solution_gpu_phase1.ipynb in VS Code Remote
# Select kernel → Python (Hydro GPU)
# Run cells!
```

**GOOD LUCK! 🚀**
