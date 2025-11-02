# Modifiche Implementate - Short_notebook_2.ipynb

## Data: 2 Novembre 2025

### 🎯 Obiettivo
Implementare le 3 modifiche con maggior impatto per migliorare lo score Kaggle da ~7,571 a ~7,000-7,100 (-471 a -571 punti).

---

## ✨ Modifiche Implementate

### 1. ⭐⭐⭐ Time-Based Cross-Validation (-80/-100 punti)

**Problema:**
- Il notebook usava `KFold` standard che mescola dati temporali
- Causava **data leakage**: usa informazioni dal futuro per predire il passato
- Le performance in cross-validation erano sovrastimate

**Soluzione Implementata:**
```python
from sklearn.model_selection import TimeSeriesSplit

def quantile_loss_cv_timeseries(model, X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        
        score = quantile_loss(y_val, y_pred, alpha=0.2)
        cv_scores.append(score)
    
    return np.mean(cv_scores)
```

**Cambiamenti:**
- ✅ Modificato `objective_catboost()` per usare `quantile_loss_cv_timeseries()`
- ✅ Modificato `objective_lgb()` per usare `quantile_loss_cv_timeseries()`
- ✅ Training samples ora ordinati per `anchor_date` prima del CV

**Impatto:**
- Previene overfitting
- CV scores più realistici
- Migliore generalizzazione su dati futuri

---

### 2. ⭐⭐⭐ Conformal Quantile Regression (-150/-200 punti)

**Problema:**
- Le predizioni quantile (α=0.2) non erano statisticamente calibrate
- Il modello poteva sistematicamente sovrastimare o sottostimare

**Soluzione Implementata:**
```python
# Split data: training (80%) + calibration (20%)
X_train_full, X_cal, y_train_full, y_cal = train_test_split(
    X_train, y_train, 
    test_size=0.2, 
    shuffle=False,  # Mantiene ordine temporale
    random_state=RANDOM_STATE
)

# Train su training set
catboost_final.fit(X_train_full, y_train_full)

# Calcola conformity scores su calibration set
y_pred_cal = catboost_final.predict(X_cal)
conformity_scores = y_pred_cal - y_cal.values

# Adjustment per quantile conservativo 0.2
alpha = 0.2
adjustment = np.quantile(conformity_scores, 1 - alpha)

# Applica alle predizioni test
pred_conformal = pred_raw - adjustment
pred_conformal = np.maximum(0, pred_conformal)
```

**Cambiamenti:**
- ✅ Aggiunto split training/calibration (80/20)
- ✅ Calcolati conformity scores per CatBoost e LightGBM
- ✅ Adjustments applicati alle predizioni finali
- ✅ Nuova variabile globale `CALIBRATION_SIZE = 0.2`

**Riferimento:**
Romano et al. "Conformalized Quantile Regression" (NeurIPS 2019)

**Impatto:**
- Predizioni quantile statisticamente garantite
- Calibrazione automatica per conservatività ottimale

---

### 3. ⭐⭐⭐ Supplier & Transportation Features (-100/-150 punti)

**Problema:**
- Non sfruttavano `supplier_id` nei receivals
- Dati `transportation.csv` ignorati
- Mancavano metriche di affidabilità fornitori, lead time, concentrazione

**Soluzione Implementata:**

Nuova funzione `engineer_supplier_features()`:

```python
def engineer_supplier_features(sample, receivals, transportation):
    rm_id = sample['rm_id']
    anchor_date = sample['anchor_date']
    features = {}
    
    # Historical deliveries last 90 days
    hist_90d = receivals[
        (receivals['rm_id'] == rm_id) &
        (receivals['arrival_date'] > anchor_date - pd.Timedelta(days=90)) &
        (receivals['arrival_date'] <= anchor_date)
    ]
    
    if len(hist_90d) > 0 and 'supplier_id' in receivals.columns:
        # Supplier diversity
        features['num_suppliers_90d'] = hist_90d['supplier_id'].nunique()
        
        # Supplier concentration (Herfindahl index)
        supplier_weights = hist_90d.groupby('supplier_id')['net_weight'].sum()
        supplier_weights = supplier_weights / supplier_weights.sum()
        features['supplier_concentration'] = (supplier_weights ** 2).sum()
        
        # Main supplier reliability
        features['main_supplier_pct'] = supplier_weights.max()
    
    # Lead time from transportation data
    if not transportation.empty:
        transp_hist = transportation[
            (transportation['rm_id'] == rm_id) &
            (transportation['batch_id'].isin(hist_90d['batch_id']))
        ]
        
        if len(transp_hist) > 0 and 'lead_time_days' in transp_hist.columns:
            features['avg_lead_time'] = transp_hist['lead_time_days'].mean()
            features['std_lead_time'] = transp_hist['lead_time_days'].std()
            features['min_lead_time'] = transp_hist['lead_time_days'].min()
            features['max_lead_time'] = transp_hist['lead_time_days'].max()
    
    return features
```

**Nuove Features (8 totali):**
1. `num_suppliers_90d` - Diversità fornitori
2. `supplier_concentration` - Indice Herfindahl (concentrazione)
3. `main_supplier_pct` - Percentuale fornitore principale
4. `avg_lead_time` - Lead time medio
5. `std_lead_time` - Deviazione standard lead time
6. `min_lead_time` - Lead time minimo
7. `max_lead_time` - Lead time massimo
8. `lead_time_days` - Calcolato in fase di caricamento dati

**Cambiamenti:**
- ✅ Aggiunta funzione `engineer_supplier_features()`
- ✅ Integrata in `engineer_enhanced_features()`
- ✅ Calcolo `lead_time_days` da `transportation.csv`
- ✅ Passaggio parametro `transportation` in tutte le chiamate feature engineering

**Impatto:**
- Pattern di consegna fornitori
- Affidabilità storica
- Previsioni più accurate per materiali con fornitori volatili

---

### 4. ⭐⭐ Material-Specific Adaptive Shrinkage (-50/-80 punti)

**Problema:**
- Shrinkage uniforme (0.93-0.99) per tutti i materiali
- Non considera volatilità e frequenza materiale

**Soluzione Implementata:**

```python
def calculate_material_shrinkage(receivals, anchor_date, base_shrinkage=0.95):
    shrinkage_by_material = {}
    
    for rm_id in receivals['rm_id'].unique():
        hist_rm = receivals[
            (receivals['rm_id'] == rm_id) &
            (receivals['arrival_date'] <= anchor_date)
        ]
        
        # Coefficient of variation (volatilità)
        cv = std_weight / (mean_weight + 1e-6)
        
        # Zero percentage
        zero_pct = (weights == 0).mean()
        
        # Frequency: deliveries per day
        frequency = len(hist_rm) / (date_range + 1)
        
        # Volatility score
        volatility_score = 0.7 * min(cv, 2.0) / 2.0 + 0.3 * zero_pct
        
        # Frequency score
        frequency_score = min(frequency / 0.1, 1.0)
        
        # Lower shrinkage = more conservative for volatile materials
        adjustment = -0.10 * volatility_score + 0.05 * frequency_score
        material_shrinkage = base_shrinkage * (1 + adjustment)
        
        # Clip to reasonable range
        shrinkage_by_material[rm_id] = np.clip(material_shrinkage, 0.80, 0.98)
    
    return shrinkage_by_material
```

**Logica:**
- **Volatile/Rare materials** → Lower shrinkage (più conservativo)
- **Stable/Frequent materials** → Higher shrinkage (meno conservativo)

**Cambiamenti:**
- ✅ Aggiunta funzione `calculate_material_shrinkage()`
- ✅ Analisi materiali con CV, frequency, zero_pct
- ✅ Array `shrinkage_factors` mappato per ogni predizione
- ✅ Shrinkage range tipico: 0.80 - 0.98

**Impatto:**
- Calibrazione fine per tipo materiale
- Meglio su materiali con pattern diversi

---

### 5. ⭐ Feature Selection (-30/-50 punti)

**Problema:**
- ~100 features generate, molte potenzialmente correlate/ridondanti
- Rischio overfitting

**Soluzione Implementata:**

```python
from sklearn.feature_selection import SelectFromModel

# Usa feature importances di CatBoost
selector = SelectFromModel(catboost_final, threshold='median', prefit=True)

X_train_selected = selector.transform(X_train_full)
X_pred_selected = selector.transform(X_pred)
```

**Cambiamenti:**
- ✅ Aggiunto import `SelectFromModel`
- ✅ Selezione top features (soglia mediana)
- ✅ Riduzione ~50% features
- ✅ Applicato sia a training che prediction data

**Impatto:**
- Riduce overfitting
- Training più veloce
- Migliore generalizzazione

---

## 📊 Submissions Generate

Il notebook ora genera 5 submission variants:

1. **`submission_conformal_material_60cat_40lgb_*.csv`** ⭐ MAIN
   - Conformal regression + Material shrinkage + 60/40 ensemble
   
2. **`submission_conformal_material_65cat_35lgb_*.csv`**
   - Più peso a CatBoost (65/35)
   
3. **`submission_conformal_material_55cat_45lgb_*.csv`**
   - Più peso a LightGBM (55/45)
   
4. **`submission_conformal_only_60cat_40lgb_*.csv`**
   - Solo conformal (senza material shrinkage)
   
5. **`submission_material_only_60cat_40lgb_*.csv`**
   - Solo material shrinkage (senza conformal)

---

## 🎯 Performance Attesa

| Scenario | Score Stimato | Miglioramento |
|----------|---------------|---------------|
| Baseline (vecchio notebook) | 7,571 | - |
| + Time-based CV | 7,490 | -81 |
| + Conformal Regression | 7,290 | -281 |
| + Supplier features | 7,140 | -431 |
| + Material shrinkage | 7,060 | -511 |
| + Feature selection | **7,030** | **-541** |
| **Target ottimistico** | **6,950 - 7,100** | **-471 a -621** |

---

## ⏱️ Runtime Stimato

- **Data loading:** ~10 secondi
- **Feature engineering (30k samples):** ~3-5 minuti
- **Optuna CatBoost (100 trials):** ~30-45 minuti
- **Optuna LightGBM (100 trials):** ~30-45 minuti
- **Training finale + calibration:** ~2-3 minuti
- **Prediction generation:** ~1-2 minuti

**Totale:** ~2-3 ore su laptop standard (4 CPU cores)

---

## 🔍 Testing Plan

### Questa Settimana (2-3 Nov)
- [x] Implementare tutte le modifiche
- [ ] Run completo del notebook
- [ ] Submit top 3 submissions a Kaggle:
  1. `submission_conformal_material_60cat_40lgb`
  2. `submission_conformal_material_65cat_35lgb`
  3. `submission_conformal_only_60cat_40lgb`
- [ ] Analizzare risultati

### Se Score Migliora
- [ ] Aumentare N_TRIALS a 200 per fine-tuning
- [ ] Testare varianti ensemble weights
- [ ] Sperimentare base_shrinkage (0.93, 0.95, 0.97)

### Se Score Non Migliora
- [ ] Verificare data leakage in features
- [ ] Debug conformity scores
- [ ] Testare submission senza feature selection

---

## 📝 Note Tecniche

### Conformal Prediction
- Adjustment calcolato su 20% calibration set
- `alpha = 0.2` per quantile conservativo
- Split temporale (shuffle=False) per rispettare ordine cronologico

### Material Shrinkage
- Base shrinkage: 0.95
- Range finale: 0.80 - 0.98
- Formula: `base * (1 + adjustment)` dove `adjustment ∈ [-0.10, +0.05]`

### Feature Selection
- Threshold: median delle feature importances
- Riduzione tipica: 40-60% features
- Mantiene features più predittive

---

## 🚀 Next Steps

1. **Immediate:**
   - Eseguire il notebook completo
   - Verificare che tutte le celle funzionino
   - Submit a Kaggle

2. **Short-term (questa settimana):**
   - Analizzare feature importances
   - Verificare calibration quality
   - Ottimizzare ensemble weights

3. **Medium-term (prossima settimana):**
   - Aumentare trials Optuna se necessario
   - Testare altri base models (XGBoost con quantile)
   - Fine-tuning finale pre-deadline

4. **Pre-Deadline (7-8 Nov):**
   - Selezionare 2 best submissions
   - Verificare reproducibilità
   - Preparare report finale

---

## ✅ Checklist Implementazione

- [x] TimeSeriesSplit implementato
- [x] Conformal quantile regression implementato
- [x] Supplier features implementate
- [x] Transportation features implementate
- [x] Material-specific shrinkage implementato
- [x] Feature selection implementato
- [x] Multiple submission strategies
- [x] Documentazione completa
- [ ] Testing completo
- [ ] Kaggle submission
- [ ] Verifica score improvement

---

## 📚 Riferimenti

1. **Conformal Prediction:**
   - Romano et al. "Conformalized Quantile Regression" (NeurIPS 2019)
   - https://proceedings.neurips.cc/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html

2. **Time Series CV:**
   - Scikit-learn TimeSeriesSplit documentation
   - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html

3. **Quantile Loss:**
   - Koenker & Bassett "Regression Quantiles" (1978)
   - Asymmetric loss function per quantile regression

---

**Autore:** Marco Prosperi  
**Data:** 2 Novembre 2025  
**Versione:** 2.0 (Advanced with Conformal Prediction)
