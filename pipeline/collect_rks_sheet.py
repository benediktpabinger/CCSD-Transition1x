"""results/rks_sheet_tzvpd.json aus den vier neuen RKS-Laeufen.

Energien woertlich aus orca_rks_sheet/<rxn>_<Modell>/ts_rks.out und
orca_om25/<rxn>_<Modell>/r_sp.out, Slurm-Job 10767516. Hartree.
"""
import io
import json

HA_EV = 27.211386245988

RAW = {
    'rxn0894/eSEN':  dict(e_rks_ts=-322.235387178769, e_r=-322.544358943374,
                          e_bs_ts=-322.381878264385, cycles=88),
    'rxn0894/UMA-M': dict(e_rks_ts=-322.223612458789, e_r=-322.544359178060,
                          e_bs_ts=-322.367167876456, cycles=84),
    'rxn8837/UMA-S': dict(e_rks_ts=-322.966513951253, e_r=-323.425092908344,
                          e_bs_ts=-323.093660103666, cycles=95),
    'rxn7060/eSEN':  dict(e_rks_ts=-323.255865354042, e_r=-323.483667629991,
                          e_bs_ts=-323.255865382619, cycles=13),
}

out = {}
for k, v in RAW.items():
    out[k] = dict(v)
    out[k]['barr'] = (v['e_rks_ts'] - v['e_r']) * HA_EV
    out[k]['barr_bs'] = (v['e_bs_ts'] - v['e_r']) * HA_EV
    out[k]['level'] = ('RKS wB97M-V/def2-TZVPD def2/J RIJCOSX TightSCF '
                       'DEFGRID3, Thresh 1e-12, TCut 1e-13, ORCA 5.0.4')
    out[k]['job'] = '10767516'

io.open('results/rks_sheet_tzvpd.json', 'w', encoding='utf-8').write(
    json.dumps(out, indent=2))
for k, v in out.items():
    print('%-15s  RKS %8.4f eV   BS %8.4f eV   Abstand der Flaechen %7.4f eV'
          % (k, v['barr'], v['barr_bs'], v['barr'] - v['barr_bs']))
