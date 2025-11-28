# W&B Local Server Setup - Komplett Offline!

## ✅ Server läuft bereits!

Der lokale W&B Server läuft auf: **http://localhost:8080**

## 📝 Setup Schritte:

### 1. Browser öffnen
```
http://localhost:8080
```

### 2. Account erstellen
- Username wählen (z.B. "local")
- Password setzen
- **KEINE Email/Internet nötig!**

### 3. API Key kopieren
- Nach Login: Settings → API Keys
- Kopiere den Key

### 4. W&B Login konfigurieren
```bash
# Setze Base URL auf lokalen Server
export WANDB_BASE_URL="http://localhost:8080"

# Login mit deinem API Key
wandb login --host=http://localhost:8080
# Dann API Key eingeben wenn gefragt
```

### 5. Offline Runs syncen
```bash
cd /Users/cedricstillecke/Documents/CloudExplain/DataScienceTutorial/Imagenette

# v6a syncen
cd v6a
wandb sync wandb/offline-run-*
cd ..

# v6b syncen
cd v6b
wandb sync wandb/offline-run-*
cd ..
```

### 6. Dashboard öffnen
```
http://localhost:8080
```

## 🎉 Was du jetzt hast:

- ✅ Vollständiges W&B Dashboard (lokal!)
- ✅ Interaktive Plots
- ✅ Run-Vergleiche (v6a vs v6b)
- ✅ Gradient Visualisierung
- ✅ Hyperparameter Vergleich
- ✅ Learning Rate Curves
- ✅ **KEIN Internet/wandb.ai benötigt!**

## 🛑 Server stoppen:
```bash
wandb server stop
```

## 🔄 Server neu starten:
```bash
wandb server start
```
