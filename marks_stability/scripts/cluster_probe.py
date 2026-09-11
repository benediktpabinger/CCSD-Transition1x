"""Probe the DTU cluster: is it reachable, and can it reach Zenodo?

Credentials come from the local memory file, never from the command line.
"""
import re
import sys
import paramiko

MEM = (r"C:\Users\PabingerBenedikt\.claude\projects"
       r"\c--Transition-1X-Transition-1x-Transition1x\memory\reference_cluster.md")

txt = open(MEM, encoding="utf-8").read()
user = re.search(r"\*\*User:\*\*\s*(\S+)", txt).group(1)
host = re.search(r"\*\*Host:\*\*\s*(\S+)", txt).group(1)
pw = re.search(r"\*\*Password:\*\*\s*(\S+)", txt).group(1)

cmds = sys.argv[1:] or [
    "hostname",
    "curl -s -m 45 -o /tmp/zen.json -w 'http=%{http_code} size=%{size_download}\\n' "
    "-L https://zenodo.org/api/records/19379882",
    "head -c 500 /tmp/zen.json",
]

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(host, username=user, password=pw, timeout=30)
for cmd in cmds:
    print("$", cmd, flush=True)
    _, out, err = c.exec_command(cmd, timeout=180)
    print(out.read().decode("utf-8", "replace")[:4000])
    e = err.read().decode("utf-8", "replace")[:1000]
    if e.strip():
        print("STDERR:", e)
c.close()
