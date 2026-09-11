"""One SSH connection to the DTU cluster, used by every helper here.

The access notes warn that a dozen rapid connections tripped a port-22 ban on
all five login nodes for over 1.5 h, so this module opens a single connection
and expects the caller to do any waiting cluster-side (one `until ...; do sleep`
loop) rather than polling from Windows.

  python cluster.py <local_script.sh> [remote_name] [timeout_s]
"""
import os
import re
import sys

import paramiko

MEM = (r"C:\Users\PabingerBenedikt\.claude\projects"
       r"\c--Transition-1X-Transition-1x-Transition1x\memory\reference_cluster.md")

DEFAULT_HOST = "thul.fysik.dtu.dk"      # slid rejects sessions
DEFAULT_USER = "s242862"
DEFAULT_KEY = r"C:\Users\PabingerBenedikt\.ssh\dtu_key"


def settings():
    """Pull host/user/key/password out of the access note, tolerating rewrites."""
    user, host, key, pw = DEFAULT_USER, DEFAULT_HOST, DEFAULT_KEY, None
    try:
        txt = open(MEM, encoding="utf-8").read()
    except OSError:
        return user, host, key, pw

    m = re.search(r"\*\*User:\*\*\s*`?([A-Za-z0-9_]+)`?", txt)
    if m:
        user = m.group(1)
    m = (re.search(r"\*\*Login node:\*\*\s*`([^`]+)`", txt)
         or re.search(r"\*\*Host:\*\*\s*`?([A-Za-z0-9_.-]+)`?", txt))
    if m:
        host = m.group(1)
    m = re.search(r"ssh\s+-i\s+(\S+)", txt)
    if m:
        p = m.group(1)
        if p.startswith("/c/"):                       # git-bash style path
            p = "C:\\" + p[3:].replace("/", "\\")
        key = p
    m = re.search(r"password\s+`([^`]+)`", txt)
    if m:
        pw = m.group(1)
    return user, host, key, pw


def connect():
    user, host, key, pw = settings()
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    if key and os.path.exists(key):
        try:
            c.connect(host, username=user, key_filename=key, timeout=30,
                      look_for_keys=False, allow_agent=False)
            # A cluster-side "until job done; do sleep" loop produces no output
            # for minutes at a time and the connection gets reaped as idle.
            c.get_transport().set_keepalive(30)
            return c, f"{user}@{host} (key)"
        except Exception as exc:                       # noqa: BLE001
            last = exc
    if pw:
        c.connect(host, username=user, password=pw, timeout=30)
        return c, f"{user}@{host} (password)"
    raise SystemExit(f"cannot connect to {host}: {last}")


def run_script(local, remote_name=None, timeout=900):
    c, how = connect()
    print(f"[{how}]", flush=True)
    remote = f"/home/energy/s242862/{remote_name or os.path.basename(local)}"
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


if __name__ == "__main__":
    run_script(sys.argv[1],
               sys.argv[2] if len(sys.argv) > 2 else None,
               int(sys.argv[3]) if len(sys.argv) > 3 else 900)
