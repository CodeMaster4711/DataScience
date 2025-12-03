# W&B Local Server - Docker Compose Setup

## 🚀 Quick Start

### 1. Server starten
```bash
cd /Users/cedricstillecke/Documents/CloudExplain/DataScienceTutorial/Imagenette

# Start W&B Server
docker-compose up -d

# Check Status
docker-compose ps

# Logs anzeigen
docker-compose logs -f wandb-local
```

### 2. Browser öffnen
```
http://localhost:8080
```

### 3. Account erstellen
- Username wählen (z.B. "codemaster4711")
- Passwort setzen
- **KEINE Email nötig!**

### 4. API Key holen
```
http://localhost:8080/authorize
```

### 5. Login konfigurieren
```bash
export WANDB_BASE_URL="http://localhost:8080"
wandb login --host=http://localhost:8080
# API Key eingeben
```

---

## 📁 Persistente Daten

Alle W&B Daten werden in **`./wandb-data/`** gespeichert:

```
wandb-data/
├── mysql/          # Datenbank
├── runs/           # Run Artefakte
├── artifacts/      # W&B Artifacts
└── media/          # Plots, Images, etc.
```

**Backup**:
```bash
# Backup erstellen
tar -czf wandb-backup-$(date +%Y%m%d).tar.gz wandb-data/

# Restore
tar -xzf wandb-backup-YYYYMMDD.tar.gz
```

---

## 🔧 Verwaltung

### Server Status
```bash
docker-compose ps
```

### Logs anzeigen
```bash
# Alle Logs
docker-compose logs -f

# Nur W&B Server
docker-compose logs -f wandb-local
```

### Server neustarten
```bash
docker-compose restart
```

### Server stoppen
```bash
# Stoppen (Daten bleiben!)
docker-compose stop

# Stoppen + Container entfernen (Daten bleiben!)
docker-compose down

# ACHTUNG: Alles löschen inkl. Volumes
docker-compose down -v  # ⚠️ LÖSCHT DATEN!
```

### Server upgraden
```bash
# Neue Version pullen
docker-compose pull

# Neu starten mit neuer Version
docker-compose up -d
```

---

## 🐍 Python Integration

### In Training Scripts
```python
import os
os.environ['WANDB_BASE_URL'] = 'http://localhost:8080'

import wandb
wandb.init(
    project="imagenette-training",
    entity="codemaster4711",  # Dein Username
    name="experiment-1"
)

# Training...
wandb.log({"loss": loss, "accuracy": acc})

wandb.finish()
```

### Offline Mode (falls Server down)
```python
import wandb
wandb.init(mode="offline", project="test")
# Läuft auch ohne Server, speichert lokal
```

---

## 🔍 Troubleshooting

### Problem: Server startet nicht
```bash
# Check Logs
docker-compose logs wandb-local

# Port 8080 schon belegt?
lsof -i :8080

# Anderen Port nutzen (docker-compose.yml ändern)
ports:
  - "8081:8080"  # 8081 statt 8080
```

### Problem: Kann mich nicht einloggen
```bash
# Server neu starten
docker-compose restart

# Browser Cache löschen
# → Incognito Mode nutzen
```

### Problem: Daten weg nach Restart
```bash
# Check ob Volume gemountet
docker-compose config | grep volumes

# Sollte zeigen: ./wandb-data:/vol

# Volume Check
ls -la wandb-data/
```

### Problem: "Permission denied"
```bash
# Fix Permissions
sudo chown -R $(whoami) wandb-data/
chmod -R 755 wandb-data/
```

---

## 📊 Dashboard Features

### Run Comparison
```
http://localhost:8080/codemaster4711/imagenette-training
```

### Parallel Coordinates
```
http://localhost:8080/codemaster4711/imagenette-training/sweeps
```

### Artifacts
```
http://localhost:8080/codemaster4711/imagenette-training/artifacts
```

---

## 🎯 Best Practices

### 1. Regelmäßige Backups
```bash
# Cronjob für tägliche Backups
0 2 * * * tar -czf ~/wandb-backup-$(date +\%Y\%m\%d).tar.gz ~/wandb-data/
```

### 2. Resource Limits setzen
```yaml
# docker-compose.yml
services:
  wandb-local:
    # ...
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

### 3. Monitoring
```bash
# Resource Usage
docker stats wandb-local

# Disk Usage
du -sh wandb-data/
```

---

## 🚫 Server komplett entfernen

```bash
# 1. Stop Container
docker-compose down

# 2. Remove Image
docker rmi wandb/local:latest

# 3. (Optional) Daten löschen
rm -rf wandb-data/

# 4. (Optional) Docker-Compose File löschen
rm docker-compose.yml
```

---

## ✅ Vorteile dieser Setup

1. **Persistent**: Daten bleiben nach Container-Restart
2. **Backup-fähig**: `wandb-data/` kann einfach gesichert werden
3. **Portable**: Gesamtes Setup in einer Datei
4. **Versioniert**: docker-compose.yml im Git
5. **Einfach**: `docker-compose up -d` startet alles

---

**Erstellt**: 2025-11-28
**Status**: ✅ Production Ready
**Version**: docker-compose v3.8
