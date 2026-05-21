import paramiko
from collections import Counter

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect("slid.fysik.dtu.dk", username="s242862", password="Butterbrot9797")

def run(cmd):
    _, out, err = ssh.exec_command(cmd)
    return out.read().decode().strip()

# --- Ground truth: count results.json files on disk ---
print("=== Results on disk ===")

# Train: 500 reactions, each should have results.json with forces
train_total   = int(run("cat ~/ccsd_dataset/train_delta_rxns.txt | wc -l"))
train_done    = int(run("find ~/train_delta_sp -name results.json | wc -l"))
train_forces  = int(run(
    "grep -rl 'forces_wb97m_eV_per_ang' ~/train_delta_sp/*/results.json 2>/dev/null | wc -l"
))
train_old     = train_done - train_forces  # energy-only (no forces key)
print(f"Train ({train_total} rxns): {train_done} results.json found "
      f"({train_forces} with forces, {train_old} energy-only/old format)")

# Val Group A: 174 reactions
val_a_total  = int(run("cat ~/ccsd_dataset/val_converged.txt | wc -l"))
val_a_done   = int(run("find ~/val_delta_sp -name results.json | wc -l"))
print(f"Val A  ({val_a_total} rxns): {val_a_done} results.json found")

# Val Group B: 51 reactions (cancelled, so likely few done)
val_b_total  = int(run("cat ~/ccsd_dataset/val_failed.txt | wc -l"))
val_b_done   = int(run("find ~/val_delta_sp_flip -name results.json | wc -l"))
val_b_forces = int(run(
    "grep -rl 'delta_forces_eV_per_ang' ~/val_delta_sp_flip/*/results.json 2>/dev/null | wc -l"
))
print(f"Val B  ({val_b_total} rxns): {val_b_done} results.json found "
      f"({val_b_forces} with forces)")

# --- squeue: what's currently running ---
print("\n=== squeue (running/pending) ===")
q = run("squeue -u s242862 --format='%.18i %.20j %.10T' -h")
counts = Counter()
for line in q.splitlines():
    parts = line.split()
    if len(parts) >= 3:
        counts[(parts[1], parts[2])] += 1
for (name, state), n in sorted(counts.items()):
    print(f"  {name:25s} {state:10s} x{n}")

import sys
_, out, _ = ssh.exec_command("tail -30 /home/energy/s242862/logs/plot_delta_10383300.log 2>&1")
print("\n=== plot_delta log (job 10383300) ===")
sys.stdout.buffer.write(out.read().replace(b'\x1b', b'').replace(b'\r', b'\n'))
print()

ssh.close()
