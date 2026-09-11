"""Cancel the oversized array and resubmit it split by system size.

The first submission asked for 12 h and 24 GB for every structure. On a
partition with ~400 queued jobs the backfill scheduler will not find a 12 h
window, so nothing started. Almost every structure here finishes in minutes
(the 10-atom control took 3), so the fast group asks for an hour and 12 GB and
backfills easily; only alanine and the 56-atom Ireland-Claisen need a long slot.

  python resubmit_sized.py [--submit]
"""
import csv
import os
import re
import sys
import paramiko

MEM = (r"C:\Users\PabingerBenedikt\.claude\projects"
       r"\c--Transition-1X-Transition-1x-Transition1x\memory\reference_cluster.md")
REMOTE = "/home/energy/s242862/marks_stab"
OLD_ARRAY = "10818211"
DONE = {"baker_06_bicyclobutane"}  # control, already complete

HERE = os.path.dirname(os.path.abspath(__file__))
INV = os.path.join(HERE, "..", "inventory.csv")

GROUPS = [
    # name,     max atoms, sbatch time, mem,  ORCA maxcore MB
    ("fast", 16, "01:00:00", "12G", 1200),
    ("slow", 10**9, "12:00:00", "48G", 5500),
]


def main():
    submit = "--submit" in sys.argv

    natoms = {}
    for row in csv.DictReader(open(INV, encoding="utf-8")):
        natoms[f"{row['set']}_{row['dir']}"] = int(row["natoms"])

    txt = open(MEM, encoding="utf-8").read()
    user = re.search(r"\*\*User:\*\*\s*(\S+)", txt).group(1)
    host = re.search(r"\*\*Host:\*\*\s*(\S+)", txt).group(1)
    pw = re.search(r"\*\*Password:\*\*\s*(\S+)", txt).group(1)

    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, username=user, password=pw, timeout=30)
    sftp = c.open_sftp()

    with sftp.open(f"{REMOTE}/tasks.txt") as fh:
        tasks = [l.split() for l in fh.read().decode().split("\n") if l.strip()]

    buckets, used = {}, set()
    for gname, maxat, _, _, _ in GROUPS:
        sel = [t for t in tasks
               if t[0] not in DONE and t[0] not in used
               and natoms.get(t[0], 0) <= maxat]
        used.update(t[0] for t in sel)
        buckets[gname] = sel

    # refresh the job script (it now honours TASKLIST and MAXCORE)
    with open(os.path.join(HERE, "job_marks_stab.sh"), "rb") as fh:
        jd = fh.read().replace(b"\r\n", b"\n")
    with sftp.open(f"{REMOTE}/job_marks_stab.sh", "wb") as fh:
        fh.write(jd)

    for gname, _, _, _, _ in GROUPS:
        sel = buckets[gname]
        if not sel:
            continue
        body = "\n".join(" ".join(t) for t in sel) + "\n"
        with sftp.open(f"{REMOTE}/tasks_{gname}.txt", "wb") as fh:
            fh.write(body.encode())
    sftp.close()

    if submit:
        out, _ = run(c, f"scancel {OLD_ARRAY}")
        print(f"cancelled array {OLD_ARRAY}")

    for gname, _, tlim, mem, maxcore in GROUPS:
        sel = buckets[gname]
        if not sel:
            continue
        print(f"\n[{gname}] {len(sel)} structures, time={tlim} mem={mem} "
              f"maxcore={maxcore}MB")
        for t in sel:
            print(f"   {t[0]:34} {natoms.get(t[0], '?'):>3} atoms")
        cmd = (f"cd {REMOTE} && sbatch --time={tlim} --mem={mem} "
               f"--array=0-{len(sel) - 1} "
               f"--export=ALL,TASKLIST={REMOTE}/tasks_{gname}.txt,MAXCORE={maxcore} "
               f"job_marks_stab.sh")
        if submit:
            o, e = run(c, cmd)
            print("  ", o.strip(), e.strip())
        else:
            print("   [dry run]", cmd)
    c.close()


def run(c, cmd):
    _, o, e = c.exec_command(cmd, timeout=120)
    return o.read().decode(), e.read().decode()


if __name__ == "__main__":
    main()
