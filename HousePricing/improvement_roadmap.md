# 🚀 Modell-Verbesserungs-Roadmap

## 📊 Aktueller Stand
- **v1 Modell:** RMSE $44,643
- **v2 Modell:** RMSE ~$39,800
- **Verbesserung:** 10.8%

---

## 🎯 Weitere Verbesserungsmöglichkeiten

### 1. ⭐ **Feature Engineering (Hoch-Priorität)**

#### 1.1 Geografisches Clustering RICHTIG implementieren
**Problem:** Aktuell ist `geo_cluster` nur ein Platzhalter (immer 0)

**Lösung:**
```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=15, random_state=42)
housing['geo_cluster'] = kmeans.fit_predict(housing[['latitude', 'longitude']])
```

**Erwartete Verbesserung:** 2-3% RMSE Reduktion

**Warum wichtig?**
- Findet automatisch "Silicon Valley", "LA Downtown", "Beach Areas"
- Jeder Cluster hat eigenes Preisniveau
- Besser als nur Koordinaten

---

#### 1.2 Zeitbasierte Features
**Problem:** `housing_median_age` ist einziges Zeit-Feature

**Neue Features:**
```python
# Altersgruppen
housing['age_category'] = pd.cut(housing['housing_median_age'],
                                  bins=[0, 10, 20, 30, 40, 50],
                                  labels=['neu', 'modern', 'mittel', 'alt', 'sehr_alt'])

# Quadrat vom Alter (nicht-lineare Beziehung)
housing['age_squared'] = housing['housing_median_age'] ** 2

# Interaktion: Alter × Einkommen
housing['age_income_interaction'] = housing['housing_median_age'] * housing['median_income']
```

**Erwartete Verbesserung:** 1-2%

---

#### 1.3 Nachbarschafts-Features
**Problem:** Jedes Haus wird isoliert betrachtet

**Lösung:**
```python
# Durchschnittspreis in der Nähe (KNN)
from sklearn.neighbors import NearestNeighbors
knn = NearestNeighbors(n_neighbors=10)
knn.fit(housing[['latitude', 'longitude']])
distances, indices = knn.kneighbors(housing[['latitude', 'longitude']])

housing['avg_price_neighborhood'] = [
    housing.iloc[idx]['median_house_value'].mean()
    for idx in indices
]
```

**Erwartete Verbesserung:** 3-5% (sehr effektiv!)

**Warum wichtig?**
- "Zeig mir deine Nachbarn, ich sage dir deinen Preis"
- Berücksichtigt lokale Preisniveaus

---

#### 1.4 Wirtschaftliche Features
**Neue Features:**
```python
# Wohlstandsindex
housing['wealth_index'] = (
    housing['median_income'] *
    housing['rooms_per_household'] *
    (1 if is_coastal else 0.8)
)

# Bevölkerungsdichte
housing['population_density'] = housing['population'] / (housing['total_rooms'] + 1)

# Wohn-Qualitäts-Score
housing['quality_score'] = (
    housing['rooms_per_household'] * 0.3 +
    housing['median_income'] * 0.5 +
    (1 if is_coastal else 0) * 0.2
)
```

**Erwartete Verbesserung:** 1-2%

---

### 2. 🔧 **Hyperparameter Tuning (Mittel-Priorität)**

#### 2.1 RandomizedSearchCV mit größerem Suchraum
**Aktuell:** GridSearch mit ~50 Kombinationen

**Verbessert:**
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_dist = {
    'model__iterations': randint(300, 1500),
    'model__depth': randint(6, 14),
    'model__learning_rate': uniform(0.01, 0.2),
    'model__l2_leaf_reg': uniform(1, 10),
    'model__subsample': uniform(0.6, 0.4),
    'model__colsample_bylevel': uniform(0.6, 0.4)
}

random_search = RandomizedSearchCV(
    pipeline, param_dist, n_iter=200, cv=5, n_jobs=-1
)
```

**Erwartete Verbesserung:** 2-4%

---

#### 2.2 Bayesian Optimization (Optuna)
**Noch besser als Random Search:**
```python
import optuna

def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 300, 1500),
        'depth': trial.suggest_int('depth', 6, 14),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10)
    }

    model = CatBoostRegressor(**params, silent=True)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
```

**Erwartete Verbesserung:** 3-6% (sehr effektiv!)

**Warum besser?**
- Lernt aus vorherigen Versuchen
- Konzentriert sich auf vielversprechende Bereiche
- Effizienter als Random Search

---

### 3. 🤝 **Ensemble Methods (Hoch-Priorität)**

#### 3.1 Stacking
**Idee:** Kombiniere mehrere Modelle

```python
from sklearn.ensemble import StackingRegressor

base_models = [
    ('catboost', CatBoostRegressor(...)),
    ('xgboost', XGBRegressor(...)),
    ('lightgbm', LGBMRegressor(...)),
    ('random_forest', RandomForestRegressor(...))
]

stacking = StackingRegressor(
    estimators=base_models,
    final_estimator=Ridge(alpha=1.0),
    cv=5
)
```

**Erwartete Verbesserung:** 3-7%

**Warum?**
- CatBoost gut bei kategorialen Features
- XGBoost gut bei Regularisierung
- LightGBM schnell und effizient
- Ridge kombiniert ihre Stärken

---

#### 3.2 Weighted Averaging
**Einfacher als Stacking:**
```python
# Trainiere 3 Modelle
pred_catboost = model1.predict(X_test)
pred_xgboost = model2.predict(X_test)
pred_lightgbm = model3.predict(X_test)

# Optimiere Gewichte (z.B. mit Grid Search)
final_pred = (
    0.5 * pred_catboost +
    0.3 * pred_xgboost +
    0.2 * pred_lightgbm
)
```

**Erwartete Verbesserung:** 2-4%

---

### 4. 🧹 **Data Cleaning & Outlier Handling**

#### 4.1 Intelligentere Outlier-Erkennung
**Aktuell:** Entfernt top 5% Fehler

**Verbessert - Isolation Forest:**
```python
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(contamination=0.05, random_state=42)
outliers = iso_forest.fit_predict(X_train)

X_train_clean = X_train[outliers == 1]
y_train_clean = y_train[outliers == 1]
```

**Erwartete Verbesserung:** 1-2%

---

#### 4.2 Besseres Imputing für fehlende Werte
**Aktuell:** SimpleImputer mit Median

**Verbessert - KNN Imputer:**
```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
X_train_imputed = imputer.fit_transform(X_train)
```

**Erwartete Verbesserung:** 1%

**Warum besser?**
- Nutzt Informationen von ähnlichen Häusern
- Realistischere Werte als einfacher Durchschnitt

---

### 5. 🎯 **Feature Selection**

#### 5.1 Entferne unwichtige Features
**Problem:** 25 Features, nicht alle wichtig

**Lösung:**
```python
from sklearn.feature_selection import SelectFromModel

# Nutze Feature Importance von CatBoost
selector = SelectFromModel(model, threshold='median')
X_train_selected = selector.fit_transform(X_train, y_train)

# Oder: RFECV (Recursive Feature Elimination)
from sklearn.feature_selection import RFECV
rfecv = RFECV(estimator=model, cv=5)
X_train_selected = rfecv.fit_transform(X_train, y_train)
```

**Erwartete Verbesserung:** 0-2% (manchmal schlechter!)

**Trade-off:**
- Weniger Features → schnelleres Training
- Kann Overfitting reduzieren
- Risiko: wichtige Features entfernt

---

### 6. 📊 **Daten-Augmentation**

#### 6.1 Synthetische Daten generieren (SMOTE für Regression)
**Idee:** Erstelle neue Trainingsbeispiele

```python
# Für Regression: Interpolation zwischen ähnlichen Häusern
from sklearn.neighbors import NearestNeighbors

def augment_data(X, y, n_augment=1000):
    knn = NearestNeighbors(n_neighbors=3)
    knn.fit(X)

    X_aug = []
    y_aug = []

    for _ in range(n_augment):
        # Wähle zufälliges Haus
        idx = np.random.randint(len(X))

        # Finde Nachbarn
        distances, indices = knn.kneighbors([X[idx]])

        # Interpoliere zwischen Haus und Nachbar
        neighbor_idx = indices[0][np.random.randint(1, 3)]
        alpha = np.random.uniform(0.3, 0.7)

        X_new = alpha * X[idx] + (1-alpha) * X[neighbor_idx]
        y_new = alpha * y[idx] + (1-alpha) * y[neighbor_idx]

        X_aug.append(X_new)
        y_aug.append(y_new)

    return np.vstack([X, X_aug]), np.hstack([y, y_aug])
```

**Erwartete Verbesserung:** 1-3%

**Vorsicht:**
- Kann Overfitting verursachen wenn zu viele
- Nur sinnvoll wenn Daten limitiert sind

---

### 7. 🧠 **Advanced Models**

#### 7.1 Neural Networks (Deep Learning)
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    Dense(128, activation='relu', input_shape=(n_features,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)  # Output
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
```

**Erwartete Verbesserung:** 0-5% (sehr variabel)

**Trade-offs:**
- Braucht VIEL Daten (funktioniert gut ab 100k+ Samples)
- Langsames Training
- Schwer zu interpretieren
- Für 20k Daten: Tree-based Modelle meist besser

---

#### 7.2 AutoML (automatisches Machine Learning)
```python
# Option 1: H2O AutoML
import h2o
from h2o.automl import H2OAutoML

h2o.init()
train = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
aml = H2OAutoML(max_runtime_secs=3600)
aml.train(x=X_train.columns.tolist(), y='median_house_value', training_frame=train)

# Option 2: PyCaret
from pycaret.regression import *
setup(data=df, target='median_house_value')
best_model = compare_models()
tuned_model = tune_model(best_model)
```

**Erwartete Verbesserung:** 5-15%

**Warum?**
- Testet automatisch viele Modelle
- Findet beste Hyperparameter
- Feature Engineering automatisch

**Nachteile:**
- Weniger Kontrolle
- Schwerer zu verstehen was passiert

---

### 8. 🔄 **Cross-Validation Strategien**

#### 8.1 Stratified K-Fold
**Problem:** Random Split kann unbalanciert sein

```python
from sklearn.model_selection import StratifiedKFold

# Erstelle Preis-Kategorien für Stratification
y_binned = pd.qcut(y, q=5, labels=False)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, test_idx in skf.split(X, y_binned):
    X_train_fold = X.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]
    # ...
```

**Erwartete Verbesserung:** 0-1% (stabilere Metriken)

---

#### 8.2 Time-Series Split (falls relevant)
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
# Nur wenn Daten zeitliche Struktur haben!
```

---

### 9. 🎨 **Target Encoding für kategoriale Features**

**Problem:** OneHotEncoding von `ocean_proximity` = 5 Features

**Besser - Target Encoding:**
```python
# Durchschnittspreis pro ocean_proximity Kategorie
ocean_means = housing.groupby('ocean_proximity')['median_house_value'].mean()

housing['ocean_proximity_encoded'] = housing['ocean_proximity'].map(ocean_means)
```

**Erwartete Verbesserung:** 1-2%

**Vorsicht:** Kann zu Data Leakage führen! Nur auf Training-Set berechnen!

---

### 10. 📈 **Target Transformation**

**Problem:** Preise sind rechts-schief (viele günstige, wenige teure)

**Lösung - Log-Transform:**
```python
# Training
y_train_log = np.log1p(y_train)
model.fit(X_train, y_train_log)

# Prediction
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)  # Zurück-transformieren
```

**Erwartete Verbesserung:** 2-5%

**Warum?**
- Reduziert Einfluss von Extremwerten
- Modell lernt besser von relativen Unterschieden
- Residuen normalverteilt → bessere Statistik

---

## 🏆 **PRIORISIERTE ROADMAP**

### Phase 1: Quick Wins (1-2 Tage)
1. ✅ Geografisches Clustering richtig implementieren (+2-3%)
2. ✅ Target Log-Transformation (+2-5%)
3. ✅ Nachbarschafts-Features (+3-5%)

**Erwartete Gesamt-Verbesserung:** 7-13%
**Neuer RMSE:** ~$35,000 - $37,000

---

### Phase 2: Hyperparameter Optimization (2-3 Tage)
4. ✅ Optuna Bayesian Optimization (+3-6%)
5. ✅ Stacking Ensemble (+3-7%)

**Erwartete Gesamt-Verbesserung:** 13-19%
**Neuer RMSE:** ~$32,000 - $35,000

---

### Phase 3: Advanced (3-5 Tage)
6. ✅ Feature Selection mit RFECV
7. ✅ AutoML (H2O oder PyCaret) (+5-15%)
8. ✅ Advanced Feature Engineering (Zeit, Wirtschaft)

**Erwartete Gesamt-Verbesserung:** 20-30%
**Neuer RMSE:** ~$28,000 - $32,000

---

### Phase 4: Experimentell (optional)
9. ⚠️ Neural Networks (unsicher)
10. ⚠️ Data Augmentation (kann schaden)

---

## 📋 **NÄCHSTER SCHRITT**

**Empfehlung: Starte mit Phase 1, Punkt 1-3**

Soll ich dir ein Notebook erstellen für:
- [ ] Geografisches Clustering + Target Log-Transform + Nachbarschafts-Features
- [ ] Optuna Hyperparameter Tuning
- [ ] Stacking Ensemble
- [ ] Alles zusammen in einem Workflow

Welches interessiert dich am meisten?
