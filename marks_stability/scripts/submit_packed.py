"""Submit the remaining structures as whole-node packed jobs on idle nodes.

xeon24el8 is saturated (one idle core), so the array sits pending. These two
partitions have a completely idle node each and only accept whole-node jobs.

  python submit_packed.py [--submit]
"""
import csv
import os
import re
import sys
import paramiko

MEM = (r"C:\Users\PabingerBenedikt\.claude\projects"
       r"\c--Transition-1X-Transition-1x-Transition1x\memory\reference_cluster.md")
REMOTE = "/home/energy/s242862/marks_stab"
HERE = os.path.dirname(os.path.abspath(__file__))

# partition, cores/node, concurrent ORCA runs, cores each, ORCA maxcore MB, time
NODES = [
    ("xeon32el9_4096", 32, 4, 8, 1800, "12:00:00"),
    ("xeon40el8_768", 40, 5, 8, 1800, "12:00:00"),
]


def run(c, cmd):
    _, o, e = c.exec_command(cmd, timeout=120)
    return o.read().decode(), e.read().decode()


def main():
    submit = "--submit" in sys.argv

    natoms = {}
    for row in csv.DictReader(open(os.path.join(HERE, "..", "inventory.csv"),
                                   encoding="utf-8")):
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

    # skip anything already finished
    done = set()
    out, _ = run(c, f"ls {REMOTE}/*/stab.out 2>/dev/null")
    for line in out.split("\n"):
        m = re.search(r"marks_stab/([^/]+)/stab.out", line)
        if m:
            chk, _ = run(c, f"grep -c 'TERMINATED NORMALLY' "
                            f"{REMOTE}/{m.group(1)}/stab.out 2>/dev/null")
            if chk.strip() not in ("", "0"):
                done.add(m.group(1))
    todo = [t for t in tasks if t[0] not in done]
    print(f"{len(done)} already complete, {len(todo)} to run")

    # biggest first so the long pole starts immediately and the node drains evenly
    todo.sort(key=lambda t: -natoms.get(t[0], 0))

    stripes = [[] for _ in NODES]
    for i, t in enumerate(todo):
        stripes[i % len(NODES)].append(t)

    with open(os.path.join(HERE, "job_marks_packed.sh"), "rb") as fh:
        jd = fh.read().replace(b"\r\n", b"\n")
    with sftp.open(f"{REMOTE}/job_marks_packed.sh", "wb") as fh:
        fh.write(jd)

    for k, (part, ncore, npar, cores, maxcore, tlim) in enumerate(NODES):
        sel = stripes[k]
        if not sel:
            continue
        name = f"tasks_pk{k}.txt"
        with sftp.open(f"{REMOTE}/{name}", "wb") as fh:
            fh.write(("\n".join(" ".join(t) for t in sel) + "\n").encode())
        print(f"\n[{part}] {len(sel)} structures, {npar} at a time x {cores} cores")
        for t in sel:
            print(f"   {t[0]:34} {natoms.get(t[0], '?'):>3} atoms")
        cmd = (f"cd {REMOTE} && sbatch --partition={part} --nodes=1 "
               f"--ntasks={ncore} --time={tlim} "
               f"--export=ALL,TASKLIST={REMOTE}/{name},NPAR={npar},"
               f"CORES={cores},MAXCORE={maxcore} job_marks_packed.sh")
        if submit:
            o, e = run(c, cmd)
            print("  ", o.strip(), e.strip())
        else:
            print("   [dry run]", cmd)

    sftp.close()
    c.close()


if __name__ == "__main__":
    main()
