# 🚀 Hydro Raw Material Forecasting - Kaggle CompetitionDescription of the files given to students



**Conservative quantile regression (0.2) for raw material delivery forecasting**README.md - this file



[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)data/kernel/receivals.csv - training data       

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)data/kernel/purchase_orders.csv - imporant data that contains whats ordered  



---data/extended/materials.csv - optional data related to the different materials            

data/extended/transportaton.csv - optional data related to transporation

## 📊 Competition Overview

data/prediction_mapping.csv - mapping used to generate submissions

Predict cumulative weight of raw material deliveries for Hydro ASA from January 1 to May 31, 2025.data/sample_submission.csv - demo submission file for Kaggle, predicting all zeros



- **Metric**: Quantile Loss at α=0.2 (asymmetric: over-prediction penalized 4x more)Dataset definitions and explanation.docx - a documents that gives more details about the dataset and column names  

- **Task**: 30,450 predictions (203 materials × ~150 time horizons)Machine learning task for TDT4173.docx - brief introduction to the task

- **Data**: 122,590 historical receivals (2004-2024), 33,171 purchase orderskaggle_metric.ipynb - the score function we use in the Kaggle competition

- **Current Best**: V4 - score 13,978 (rank: 43rd)
- **Target**: TOP 10 (score < 6,000) 🎯

---

## 🏆 Results Summary

| Version | Model | Features | Score | Rank | Improvement |
|---------|-------|----------|-------|------|-------------|
| V1 | Heuristic | 5 | ~35,000 | - | Baseline |
| V3 | LightGBM + XGBoost | 20 | 40,032 | - | ML baseline |
| **V4** | **LightGBM + Optuna** | **33** | **37,917 (val)** | - | **5.3%** ✅ |
| **V4.2** | **Same (no constraint)** | **33** | **13,978** | **43rd** | **Current best** 🏆 |
| V5 | Ultra-conservative | 33 | TBD | - | Too extreme |
| **GPU** | **CatBoost + XGBoost + TFT** | **100+** | **TBD** | **TBD** | **Target: TOP 10** 🎯 |

---

## 🚀 Quick Start

### Option 1: Local Execution (CPU)

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/hydro-forecasting-kaggle.git
cd hydro-forecasting-kaggle

# Create environment
conda create -n hydro python=3.10 -y
conda activate hydro

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn
pip install lightgbm xgboost optuna

# Open best model
jupyter notebook solution_ml_tuned.ipynb
```

### Option 2: GPU Execution (Recommended for TOP 10)

See **[GITHUB_GPU_SETUP_GUIDE.md](GITHUB_GPU_SETUP_GUIDE.md)** for complete instructions.

**Quick setup:**

```bash
# 1. Initialize git (local)
bash setup.sh init-git

# 2. Create GitHub repo
gh repo create hydro-forecasting-kaggle --private --source=. --push

# 3. On GPU server: setup
bash setup.sh setup-gpu
conda activate hydro
bash setup.sh install-deps

# 4. Run GPU notebook
jupyter notebook solution_gpu_phase1.ipynb
```

---

## 📁 Project Structure

```
├── solution_ml_tuned.ipynb           # 🏆 BEST CURRENT MODEL (V4)
├── solution_gpu_phase1.ipynb         # 🚀 GPU Quick Wins (Phase 1)
├── solution.ipynb                    # V1 Heuristic baseline
├── solution_v2.ipynb                 # V2 Statistical
├── solution_ml.ipynb                 # V3 ML baseline
├── solution_v5_ultra_conservative.ipynb  # V5-V7 experiments
│
├── submissions/
│   ├── submission_ml_tuned_v2.csv    # 🏆 Best (13,978 score)
│   └── submission_gpu_*.csv          # GPU predictions
│
├── data/
│   ├── kernel/
│   │   ├── receivals.csv             # Training data (122,590 rows)
│   │   └── purchase_orders.csv       # Future orders (33,171 rows)
│   ├── extended/
│   │   ├── materials.csv             # Material metadata
│   │   └── transportation.csv        # Transportation data
│   └── prediction_mapping.csv        # Competition mapping (30,450 predictions)
│
├── GPU_STRATEGY.md                   # 📖 Complete GPU strategy
├── IMPLEMENTATION_ROADMAP.md         # 🗺️ 7-12 day implementation plan
├── GITHUB_GPU_SETUP_GUIDE.md         # 🛠️ GitHub + VS Code Remote setup
├── setup.sh                          # 🔧 Automation script
└── README.md                         # This file
```

---

## 🧠 Best Model Architecture (V4)

### Features (33)

**Historical Statistics:**
- mean, std, p10, p20, p50, CV, skewness, kurtosis

**Rolling Windows:**
- 30d, 60d, 90d: mean, p10, p20, count, daily rate, trend

**Delivery Patterns:**
- Frequency, avg days between deliveries, days since last

**Purchase Orders:**
- Future orders count, weight, average

**Seasonality:**
- Month/quarter sin/cos encoding

**Trends:**
- Linear regression on recent windows
- Acceleration (2nd derivative)

### Hyperparameters (Optuna optimized, 20 trials)

```python
{
    'objective': 'quantile',
    'alpha': 0.2,
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'min_data_in_leaf': 20,
    'lambda_l1': 1.0,
    'lambda_l2': 1.0,
    'max_depth': -1,
    'min_gain_to_split': 0.01
}
```

### Performance

- **Validation QL**: 37,917 (5.3% better than baseline)
- **Under-predictions**: 75.4% (target: ~80%)
- **Mean prediction**: 66,989 kg
- **Zero predictions**: 15.8%
- **Kaggle score**: 13,978 (43rd place)

---

## 🔮 GPU Strategy - Path to TOP 10

### Phase 1: Quick Wins (1-2 days) → 25-35% improvement

**File:** `solution_gpu_phase1.ipynb`

**Enhancements:**
- **CatBoost GPU** (10,000 iterations vs 500)
- **XGBoost GPU** (3,000 iterations)
- **100+ advanced features**:
  - Multiple rolling windows (7, 14, 30, 60, 90, 180, 365d)
  - 9 quantile levels (p05, p10, p15, p20, p25, p30, p50, p75, p90)
  - Lag features (1, 7, 14, 30, 60, 90d)
  - Momentum & acceleration
  - Interaction features (horizon × freq, horizon × trend, etc.)
- **Ensemble** (weighted average)

**Expected:** Score 9,000-10,500 (~20th place)

### Phase 2: Deep Learning (3-5 days) → +15-25% improvement

**Models:**
- Temporal Fusion Transformer (attention-based time series)
- TabNet (attention-based tabular)
- LSTM/GRU (recurrent networks)

**Expected:** Score 6,500-8,000 (~15th place)

### Phase 3: Advanced Ensemble (2-3 days) → +10-15% improvement

- 3-layer stacking (6 base models → meta-model → blending)
- Uncertainty quantification

**Expected:** Score 5,500-7,000 (~10th place)

### Phase 4: Massive Tuning (1-2 days) → +5-8% improvement

- Optuna 1000+ trials per model (vs 20 currently)
- Ensemble weight optimization

**Expected:** Score 5,000-6,500 (TOP 5-15!) 🏆

**See [GPU_STRATEGY.md](GPU_STRATEGY.md) for details.**

---

## 📊 Key Insights

### Why Quantile 0.2 is Challenging

Quantile Loss is **asymmetric**:

```
QL = Σ (0.2 × error)      if y_true ≥ y_pred  (under-prediction, weight 0.2)
     (0.8 × |error|)      if y_true < y_pred  (over-prediction, weight 0.8)
```

**Over-predicting is 4x worse than under-predicting!**

**Strategy:**
- Target ~80% under-predictions
- Be very conservative for:
  - Materials with sparse history
  - Short forecast horizons (1-30 days)
  - Obsolete materials (no recent deliveries)

### Model Learned Behaviors

**Correct patterns:**
- Predicts **zero** for materials without recent deliveries
- Lower predictions for short horizons
- Higher predictions for longer horizons
- Seasonal adjustments (sin/cos month encoding)
- Purchase order integration helps significantly

**Example:** rm_id 365
- No deliveries since 2005
- Model correctly predicts zero for Jan 2025
- This is the **right** conservative choice!

---

## 🛠️ Tools & Technologies

**Core:**
- Python 3.10+
- pandas, numpy, scikit-learn

**Machine Learning:**
- LightGBM (current best)
- XGBoost
- CatBoost (GPU)

**Deep Learning (GPU):**
- PyTorch 2.0+
- PyTorch Forecasting (TFT)
- PyTorch TabNet

**Optimization:**
- Optuna (Bayesian hyperparameter tuning)

---

## 🔧 Setup Commands

### Helper Script

```bash
# Local: Initialize git
bash setup.sh init-git

# Local: Sync data to GPU server
bash setup.sh sync-data gpu-server

# Remote: Setup GPU environment
bash setup.sh setup-gpu
conda activate hydro
bash setup.sh install-deps

# Remote: Monitor GPU
bash setup.sh monitor-gpu

# Local: Download results
bash setup.sh download-results gpu-server
```

---

## 📈 Monitoring

### GPU Usage

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Or with gpustat
pip install gpustat
gpustat -i 1
```

### Model Validation

```python
def quantile_loss(y_true, y_pred, quantile=0.2):
    errors = y_true - y_pred
    loss = np.where(errors >= 0, 
                   quantile * errors, 
                   (quantile - 1) * errors)
    return loss.sum()

# Check under-prediction ratio
under_preds = np.sum(preds < y_true) / len(y_true)
print(f"Under-predictions: {under_preds:.1%}")  # Target: ~80%
```

---

## 📚 Documentation

- **[GPU_STRATEGY.md](GPU_STRATEGY.md)** - Detailed GPU implementation strategy
  - Temporal Fusion Transformer
  - CatBoost/XGBoost GPU
  - Feature engineering (100+ features)
  - 3-layer stacking ensemble
  - Massive Optuna tuning

- **[IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md)** - Step-by-step plan
  - 4 phases over 7-12 days
  - Expected improvements per phase
  - Time estimates & priorities

- **[GITHUB_GPU_SETUP_GUIDE.md](GITHUB_GPU_SETUP_GUIDE.md)** - Complete setup
  - Create GitHub repo
  - VS Code Remote SSH
  - GPU server configuration
  - Data synchronization

---

## 🎯 Next Steps

### Immediate (Today)

1. **Setup GitHub repo**
   ```bash
   bash setup.sh init-git
   gh repo create hydro-forecasting-kaggle --private --source=. --push
   ```

2. **Access GPU server** (Kaggle, Lambda Labs, RunPod, etc.)

3. **Clone & setup**
   ```bash
   git clone https://github.com/YOUR_USERNAME/hydro-forecasting-kaggle.git
   cd hydro-forecasting-kaggle
   bash setup.sh setup-gpu
   conda activate hydro
   bash setup.sh install-deps
   ```

### Short-term (1-2 days)

4. **Execute Phase 1**
   - Open `solution_gpu_phase1.ipynb`
   - Run all cells (~1 hour)
   - Upload `submission_gpu_ensemble.csv`
   - **Expected: Score 9,000-10,500** ✅

### Medium-term (1-2 weeks)

5. **Execute Phases 2-4**
   - Implement TFT, TabNet
   - Build stacking ensemble
   - Massive Optuna tuning
   - **Target: TOP 10 (score < 6,000)** 🏆

---

## 🏅 Success Criteria

| Target | Score Range | Rank | Status |
|--------|-------------|------|--------|
| Minimum | < 10,000 | TOP 25 | ✅ Achievable with Phase 1 |
| Good | < 8,000 | TOP 15 | ✅ Achievable with Phase 2 |
| Great | < 6,000 | TOP 10 | 🎯 Target with Phases 3-4 |
| Excellent | < 5,000 | TOP 5 | 🏆 Stretch goal |

**Current:** 13,978 (43rd) → **Next:** 9,000-10,500 (~20th) → **Final:** 5,000-6,500 (TOP 5-15!)

---

## 💡 Tips

1. **Start with Phase 1** - Quick wins, immediate improvement
2. **Monitor GPU usage** - Use `nvidia-smi` or `gpustat`
3. **Save checkpoints** - Don't lose 10 hours of training
4. **Cross-validate** - Use 10-fold for robustness
5. **Try multiple submissions** - Different ensembles, different scales
6. **Scale predictions** - If too high, multiply by 0.8-0.9

---

## 🙏 Acknowledgments

- Hydro ASA for the dataset
- Kaggle for hosting
- PyTorch Forecasting & Optuna teams

---

**Good luck! 🚀 Target: TOP 10! 🏆**
