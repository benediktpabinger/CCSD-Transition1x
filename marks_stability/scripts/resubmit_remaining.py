"""Cancel outstanding packed jobs, rebuild the to-do list from what actually
completed, and resubmit with the fixed job script.

  python resubmit_remaining.py [--submit] [--partitions p1:ntasks:npar,...]

"Completed" means ORCA terminated normally: the stability run for a closed-shell
singlet, the reference run for an open-shell structure. Anything else is redone.
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

DEFAULT_PARTS = [("xeon40el8_768", 40, 5)]


def run(c, cmd, t=180):
    _, o, e = c.exec_command(cmd, timeout=t)
    return o.read().decode(), e.read().decode()


def main():
    submit = "--submit" in sys.argv
    parts = DEFAULT_PARTS
    if "--partitions" in sys.argv:
        spec = sys.argv[sys.argv.index("--partitions") + 1]
        parts = []
        for item in spec.split(","):
            p, n, k = item.split(":")
            parts.append((p, int(n), int(k)))

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

    if submit:
        out, _ = run(c, "squeue -h -u s242862 -n marksPk,marksstab -o '%A'")
        ids = sorted({x.strip() for x in out.split() if x.strip()})
        if ids:
            run(c, "scancel " + " ".join(ids))
            print("cancelled:", " ".join(ids))
        run(c, "sleep 3")

    sftp = c.open_sftp()
    with sftp.open(f"{REMOTE}/tasks.txt") as fh:
        tasks = [l.split() for l in fh.read().decode().split("\n") if l.strip()]

    # one remote pass: report which structures have a genuinely finished run
    probe = []
    for rid, chg, mult, geom in tasks:
        f = "stab.out" if mult == "1" else "ref.out"
        probe.append(f"grep -lq 'ORCA TERMINATED NORMALLY' {REMOTE}/{rid}/{f} "
                     f"2>/dev/null && echo DONE {rid} || echo TODO {rid}")
    out, _ = run(c, "; ".join(probe), t=300)
    done = {l.split()[1] for l in out.split("\n") if l.startswith("DONE")}

    todo = [t for t in tasks if t[0] not in done]
    print(f"{len(done)} complete, {len(todo)} to (re)run")

    # clear partial directories so a rerun is clean
    if submit and todo:
        run(c, "; ".join(f"rm -rf {REMOTE}/{t[0]}" for t in todo), t=300)

    todo.sort(key=lambda t: -natoms.get(t[0], 0))
    stripes = [[] for _ in parts]
    for i, t in enumerate(todo):
        stripes[i % len(parts)].append(t)

    with open(os.path.join(HERE, "job_marks_packed.sh"), "rb") as fh:
        jd = fh.read().replace(b"\r\n", b"\n")
    with sftp.open(f"{REMOTE}/job_marks_packed.sh", "wb") as fh:
        fh.write(jd)

    for k, (part, ntasks, npar) in enumerate(parts):
        sel = stripes[k]
        if not sel:
            continue
        name = f"tasks_r{k}.txt"
        with sftp.open(f"{REMOTE}/{name}", "wb") as fh:
            fh.write(("\n".join(" ".join(t) for t in sel) + "\n").encode())
        print(f"\n[{part}] {len(sel)} structures, {npar} concurrent x 8 cores")
        for t in sel:
            print(f"   {t[0]:34} {natoms.get(t[0], '?'):>3} atoms")
        cmd = (f"cd {REMOTE} && sbatch --partition={part} --nodes=1 "
               f"--ntasks={ntasks} --time=12:00:00 "
               f"--export=ALL,TASKLIST={REMOTE}/{name},NPAR={npar},CORES=8,"
               f"MAXCORE=1800 job_marks_packed.sh")
        if submit:
            o, e = run(c, cmd)
            print("  ", o.strip(), e.strip())
        else:
            print("   [dry run]", cmd)
    sftp.close()
    c.close()


if __name__ == "__main__":
    main()
