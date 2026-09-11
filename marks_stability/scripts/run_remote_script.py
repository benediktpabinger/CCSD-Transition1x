"""Upload a local shell script to the cluster and run it there."""
import re
import sys
import paramiko

MEM = (r"C:\Users\PabingerBenedikt\.claude\projects"
       r"\c--Transition-1X-Transition-1x-Transition1x\memory\reference_cluster.md")

txt = open(MEM, encoding="utf-8").read()
user = re.search(r"\*\*User:\*\*\s*(\S+)", txt).group(1)
host = re.search(r"\*\*Host:\*\*\s*(\S+)", txt).group(1)
pw = re.search(r"\*\*Password:\*\*\s*(\S+)", txt).group(1)

local = sys.argv[1]
remote = sys.argv[2]
timeout = int(sys.argv[3]) if len(sys.argv) > 3 else 600

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(host, username=user, password=pw, timeout=30)

sftp = c.open_sftp()
with open(local, "rb") as fh:
    data = fh.read().replace(b"\r\n", b"\n")
with sftp.open(remote, "wb") as fh:
    fh.write(data)
sftp.chmod(remote, 0o755)
sftp.close()

_, out, err = c.exec_command(f"bash {remote}", timeout=timeout)
print(out.read().decode("utf-8", "replace"))
e = err.read().decode("utf-8", "replace")
if e.strip():
    print("STDERR:", e[:3000])
c.close()
