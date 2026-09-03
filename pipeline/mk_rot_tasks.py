"""Taskliste fuer den Rotations-Gegentest: rxn:Modell:HOMO:LUMO:NEL

HOMO = n_elec/2 - 1, LUMO = n_elec/2, n_elec aus fod_ranking.json. Die Formel
ist gegen die von Hand gepflegten Arrays in job_omol25_settings.sh geprueft
(26 von 26 identisch); diese Pruefung laeuft hier noch einmal mit.
"""
import csv
import json

H = '/home/energy/s242862'
LONG = {'uma-s': 'UMA-S', 'uma-m': 'UMA-M', 'esen': 'eSEN'}

res = {r['rxn']: r for r in json.load(open(H + '/fod_ranking.json'))['results']}

# Gegenprobe gegen die handgepflegten Indizes des Validierungslaufs
RX = ('rxn7949 rxn8832 rxn1320 rxn4113 rxn8885 rxn7945 rxn7937 rxn6196 rxn0346 '
      'rxn1150 rxn0896 rxn4518 rxn3107 rxn8837 rxn7060 rxn5691 rxn1283 rxn8827 '
      'rxn4522 rxn7936 rxn1147 rxn0894 rxn0101 rxn10005 rxn10054 rxn7957').split()
HO = [24, 24, 22, 22, 24, 24, 24, 24, 22, 22, 22, 22, 22, 24, 24, 24, 22, 24,
      22, 24, 22, 22, 22, 25, 24, 24]
for rx, h in zip(RX, HO):
    ne = res[rx]['n_elec']
    assert ne // 2 - 1 == h, '%s: n_elec=%d -> %d, Skript sagt %d' % (
        rx, ne, ne // 2 - 1, h)
print('Formelpruefung gegen job_omol25_settings.sh: %d von %d identisch'
      % (len(RX), len(RX)))

rows = list(csv.DictReader(open(H + '/results/omol25_model_geoms.csv')))
lines, miss = [], []
for r in rows:
    rx = r['rxn']
    if rx not in res or res[rx].get('n_elec') is None:
        miss.append(rx)
        continue
    ne = res[rx]['n_elec']
    assert ne % 2 == 0, '%s: ungerade Elektronenzahl %d' % (rx, ne)
    lines.append('%s:%s:%d:%d:%d' % (rx, LONG[r['model']], ne // 2 - 1,
                                     ne // 2, ne))

assert not miss, 'kein n_elec fuer: %s' % sorted(set(miss))
assert len(lines) == 135, 'erwartet 135 Aufgaben, erzeugt %d' % len(lines)
open(H + '/rot_check_tasks.txt', 'w').write('\n'.join(lines) + '\n')
print('geschrieben: rot_check_tasks.txt (%d Aufgaben)' % len(lines))
print('erste drei:', lines[:3])
