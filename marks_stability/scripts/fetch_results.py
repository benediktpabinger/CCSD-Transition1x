"""Archive the ORCA outputs on the cluster and pull them down, in one session.

Keeps the inputs, the .out files and the .engrad (the gradient check); leaves
the scratch (.gbw, .tmp, integrals) behind.

  python fetch_results.py <local_dest_dir>
"""
import os
import sys
import tarfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cluster import connect  # noqa: E402

REMOTE = "/home/energy/s242862/marks_stab"
TAR = "/home/energy/s242862/marks_stab_outputs.tar.gz"

dest = sys.argv[1]
os.makedirs(dest, exist_ok=True)

c, how = connect()
print(f"[{how}]")

cmd = (f"cd {REMOTE} && tar czf {TAR} "
       f"--exclude='*.gbw' --exclude='*.tmp*' --exclude='*.bas*' "
       f"*/ref.out */ref.err */ref.inp */ref.engrad */stab.out */stab.err "
       f"*/stab.inp */start.xyz */uks.out */uks.inp tasks*.txt packed_*.out packed_*.err "
       f"slurm_*.out 2>/dev/null; ls -l {TAR}")
_, out, err = c.exec_command(cmd, timeout=600)
print(out.read().decode())
e = err.read().decode()
if e.strip():
    print("STDERR:", e[:800])

local_tar = os.path.join(dest, "marks_stab_outputs.tar.gz")
sftp = c.open_sftp()
sftp.get(TAR, local_tar)
sftp.close()
c.close()

print(f"downloaded {os.path.getsize(local_tar) / 1e6:.1f} MB -> {local_tar}")
with tarfile.open(local_tar) as t:
    t.extractall(dest)
n = sum(1 for _, _, fs in os.walk(dest) for f in fs if f.endswith(".out"))
print(f"extracted; {n} .out files under {dest}")
