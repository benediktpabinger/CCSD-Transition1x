import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('slid.fysik.dtu.dk', username='s242862', password='Butterbrot9797')
def run(cmd):
    _, out, _ = ssh.exec_command(cmd, get_pty=True)
    return out.read().decode('utf-8', errors='replace').strip()

sftp = ssh.open_sftp()
script = (
    "import ase.db, numpy as np\n"
    "db = ase.db.connect('/home/energy/s242862/orca_neb_results/rxn7949/neb.db')\n"
    "rows = list(db.select())\n"
    "row = rows[-1]\n"
    "atoms = row.toatoms()\n"
    "print('energy:', atoms.get_potential_energy())\n"
    "try:\n"
    "    f = atoms.get_forces()\n"
    "    print('forces shape:', f.shape)\n"
    "    print('forces[0]:', f[0])\n"
    "except Exception as e:\n"
    "    print('forces error:', e)\n"
    "print('keys in row:', list(row.key_value_pairs.keys()))\n"
    "print('data keys:', list(row.data.keys()) if row.data else 'none')\n"
)
with sftp.open('/home/energy/s242862/pipeline/_check_neb_forces.py', 'w') as f:
    f.write(script)
sftp.close()

print(run('source /etc/profile; module load Python/3.13.5-GCCcore-14.3.0 && python3 /home/energy/s242862/pipeline/_check_neb_forces.py 2>&1'))
ssh.close()
