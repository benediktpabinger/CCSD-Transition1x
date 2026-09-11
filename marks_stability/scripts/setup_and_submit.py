"""Build the task list, upload geometries, and submit the ORCA stability jobs.

  python setup_and_submit.py <repo_geoms_dir> [--only NAME[,NAME...]] [--submit]

Without --submit it only writes and uploads, printing the sbatch line it would
run. The 28 distinct structures are Baker 1-24 plus Sharada 02/07/08/09; the
other five Sharada reference TSs are byte-identical (or identical to <0.002 A)
to their Baker twins and are mapped back in the analysis instead of recomputed.
"""
import posixpath
import re
import sys
import paramiko

MEM = (r"C:\Users\PabingerBenedikt\.claude\projects"
       r"\c--Transition-1X-Transition-1x-Transition1x\memory\reference_cluster.md")
REMOTE = "/home/energy/s242862/marks_stab"

BAKER = ["01_hcn", "02_hcch", "03_h2co", "04_ch3o", "05_cyclopropyl",
         "06_bicyclobutane", "08_formyloxyethyl", "09_parentdielsalder",
         "10_tetrazine", "11_trans_butadiene", "12_ethane_h2_abstraction",
         "13_hf_abstraction", "14_vinyl_alcohol", "15_hocl", "16_h2po4_anion",
         "17_claisen", "18_silylene_insertion", "19_hnccs", "20_hconh3_cation",
         "21_acrolein_rot", "22_hconhoh", "23_hcn_h2", "24_h2cnh", "25_hcnh2"]
SHARADA = ["02_silane", "07_hexadiene", "08_alanine", "09_icr"]

# sharada/04 ships chg=1 mult=0 -- unphysical, the two files are swapped. It is
# a duplicate of baker/12 and not submitted, but the guard stays in case the
# selection ever changes.
FIXES = {("sharada", "04_ethane_dehydrogenation"): (0, 1)}


def read_meta(root, setname, d):
    base = f"{root}/data/{setname}/{d}"
    chg = int(open(f"{base}/chg").read().split()[0])
    mult = int(open(f"{base}/mult").read().split()[0])
    if (setname, d) in FIXES:
        chg, mult = FIXES[(setname, d)]
    return chg, mult, f"{base}/ts.xyz"


def main():
    root = sys.argv[1].replace("\\", "/")
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    submit = "--submit" in sys.argv

    tasks = []
    for setname, dirs in (("baker", BAKER), ("sharada", SHARADA)):
        for d in dirs:
            rid = f"{setname}_{d}"
            if only and rid not in only:
                continue
            chg, mult, path = read_meta(root, setname, d)
            tasks.append((rid, chg, mult, path))

    if not tasks:
        sys.exit("no tasks selected")

    txt = open(MEM, encoding="utf-8").read()
    user = re.search(r"\*\*User:\*\*\s*(\S+)", txt).group(1)
    host = re.search(r"\*\*Host:\*\*\s*(\S+)", txt).group(1)
    pw = re.search(r"\*\*Password:\*\*\s*(\S+)", txt).group(1)

    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, username=user, password=pw, timeout=30)
    sftp = c.open_sftp()

    def mkdir(p):
        try:
            sftp.mkdir(p)
        except IOError:
            pass

    mkdir(REMOTE)
    mkdir(f"{REMOTE}/geoms")

    lines = []
    for rid, chg, mult, path in tasks:
        rpath = f"{REMOTE}/geoms/{rid}.xyz"
        with open(path, "rb") as fh:
            data = fh.read().replace(b"\r\n", b"\n")
        with sftp.open(rpath, "wb") as fh:
            fh.write(data)
        lines.append(f"{rid} {chg} {mult} {rpath}")

    with sftp.open(f"{REMOTE}/tasks.txt", "wb") as fh:
        fh.write(("\n".join(lines) + "\n").encode())

    job = sys.argv[0].replace("setup_and_submit.py", "job_marks_stab.sh")
    with open(job, "rb") as fh:
        jd = fh.read().replace(b"\r\n", b"\n")
    with sftp.open(f"{REMOTE}/job_marks_stab.sh", "wb") as fh:
        fh.write(jd)
    sftp.close()

    print(f"uploaded {len(tasks)} geometries + tasks.txt to {REMOTE}")
    for ln in lines:
        print("  ", ln)

    cmd = f"cd {REMOTE} && sbatch --array=0-{len(tasks) - 1} job_marks_stab.sh"
    if not submit:
        print(f"\n[dry run] would run: {cmd}")
    else:
        _, out, err = c.exec_command(cmd, timeout=120)
        print(out.read().decode(), err.read().decode())
    c.close()


if __name__ == "__main__":
    main()
