# CCSD Test Script für Transition1x

## Verwendung

### 1. Verfügbare Reaktionen anzeigen

```bash
python ccsd_test.py --list-reactions
```

### 2. Lokal auf Login-Node testen (nur 1-2 Konfigurationen!)

```bash
python ccsd_test.py --reaction "C2H4+H" --max-configs 2 --basis cc-pVDZ
```

### 3. Als SLURM-Job einreichen (empfohlen)

**Wichtig:** Bearbeiten Sie zuerst `run_ccsd_test.sh` und passen Sie die Reaktion an!

```bash
# Job einreichen
sbatch run_ccsd_test.sh

# Job-Status prüfen
squeue -u $USER

# Log-Datei ansehen
tail -f ccsd_test_*.log
```

## Script-Optionen

```bash
python ccsd_test.py --help
```

### Wichtige Parameter:

- `--reaction`: Name der Reaktion (z.B. "C2H4+H")
- `--max-configs`: Limitiere Anzahl der Berechnungen (für Tests)
- `--basis`: Basissatz (cc-pVDZ, cc-pVTZ, aug-cc-pVDZ)
- `--split`: train/val/test (default: test)
- `--h5file`: Pfad zur HDF5-Datei

## Beispiele

### Eine Reaktion komplett berechnen:
```bash
python ccsd_test.py --reaction "C2H4+H"
```

### Test mit nur 3 Konfigurationen:
```bash
python ccsd_test.py --reaction "CH4+O" --max-configs 3
```

### Mit größerem Basissatz:
```bash
python ccsd_test.py --reaction "C2H4+H" --basis cc-pVTZ --max-configs 5
```

## Module auf Niflheim

Verfügbare Quantum Chemistry Module finden:
```bash
module spider Psi4
module spider ORCA
module spider NWChem
```

GPAW/ASE Module:
```bash
module avail GPAW
module avail ASE
```

## Output

Ergebnisse werden gespeichert in:
- `ccsd_<reaktionsname>/` - Berechnungsverzeichnis
- `ccsd_<reaktionsname>/ccsd_results.json` - JSON mit allen Ergebnissen

## Wichtige Hinweise

1. **CCSD ist sehr rechenintensiv** - beginnen Sie mit wenigen Konfigurationen zum Testen
2. **Login-Nodes nicht überlasten** - verwenden Sie SLURM-Jobs für echte Berechnungen
3. **Basissatz-Wahl**: 
   - cc-pVDZ: Schnell, für Tests
   - cc-pVTZ: Höhere Genauigkeit, viel langsamer
   - aug-cc-pVDZ: Mit diffusen Funktionen

4. **Rechenzeit**: 
   - Eine CCSD-Berechnung kann 10 Minuten bis mehrere Stunden dauern
   - Abhängig von Molekülgröße und Basissatz

## Troubleshooting

### "No module named 'psi4'":
```bash
module load Psi4/1.9.1-foss-2024a
```

### "No module named 'transition1x'":
```bash
cd /path/to/Transition1x
pip install --user .
```

### Calculation failed:
- Prüfen Sie genug Speicher (--mem in SLURM)
- Kleinerer Basissatz für große Moleküle
- Prüfen Sie die Log-Dateien
