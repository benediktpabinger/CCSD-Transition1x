"""Upload the packed job script and submit one existing remote task list.

  python submit_list.py <tasks_file> <partition> <ntasks> <npar> [--submit]

Used to requeue the batch that died on the el9 node onto an el8 partition,
where the ORCA/5.0.4-gompi-2023a module actually exists.
"""
import os
import re
import sys
import paramiko

MEM = (r"C:\Users\PabingerBenedikt\.claude\projects"
       r"\c--Transition-1X-Transition-1x-Transition1x\memory\reference_cluster.md")
REMOTE = "/home/energy/s242862/marks_stab"
HERE = os.path.dirname(os.path.abspath(__file__))

tasks, part, ntasks, npar = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
submit = "--submit" in sys.argv

txt = open(MEM, encoding="utf-8").read()
user = re.search(r"\*\*User:\*\*\s*(\S+)", txt).group(1)
host = re.search(r"\*\*Host:\*\*\s*(\S+)", txt).group(1)
pw = re.search(r"\*\*Password:\*\*\s*(\S+)", txt).group(1)

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(host, username=user, password=pw, timeout=30)

sftp = c.open_sftp()
with open(os.path.join(HERE, "job_marks_packed.sh"), "rb") as fh:
    jd = fh.read().replace(b"\r\n", b"\n")
with sftp.open(f"{REMOTE}/job_marks_packed.sh", "wb") as fh:
    fh.write(jd)
with sftp.open(f"{REMOTE}/{tasks}") as fh:
    n = len([l for l in fh.read().decode().split("\n") if l.strip()])
sftp.close()

cmd = (f"cd {REMOTE} && sbatch --partition={part} --nodes=1 --ntasks={ntasks} "
       f"--time=12:00:00 "
       f"--export=ALL,TASKLIST={REMOTE}/{tasks},NPAR={npar},CORES=8,MAXCORE=1800 "
       f"job_marks_packed.sh")
print(f"{tasks}: {n} structures -> {part}, {npar} concurrent")
if submit:
    _, o, e = c.exec_command(cmd, timeout=120)
    print(o.read().decode().strip(), e.read().decode().strip())
else:
    print("[dry run]", cmd)
c.close()
