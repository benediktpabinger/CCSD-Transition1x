"""
Upload locally-edited MACE files to the cluster, overwriting the installed package.

Usage:
    python deploy_mace_patches.py

Edit pipeline/mace/arg_parser.py and pipeline/mace/run_train.py locally,
then run this script to deploy them.
"""

import os

import paramiko
import pathlib

CLUSTER_HOST = "slid.fysik.dtu.dk"
CLUSTER_USER = "s242862"
CLUSTER_KEY  = os.path.expanduser("~/.ssh/dtu_key")

MACE_SITE = "/home/energy/s242862/.local/lib/python3.13/site-packages/mace"

FILES = {
    "mace/arg_parser.py": f"{MACE_SITE}/tools/arg_parser.py",
    "mace/run_train.py":  f"{MACE_SITE}/cli/run_train.py",
}

LOCAL_BASE = pathlib.Path(__file__).parent


def deploy():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    key = paramiko.Ed25519Key.from_private_key_file(CLUSTER_KEY)
    ssh.connect(CLUSTER_HOST, username=CLUSTER_USER, pkey=key)
    sftp = ssh.open_sftp()

    for local_rel, remote_path in FILES.items():
        local_path = LOCAL_BASE / local_rel
        assert local_path.exists(), f"Not found: {local_path}"
        sftp.put(str(local_path), remote_path)
        print(f"Uploaded: {local_rel} -> {remote_path}")

    # Clear stale pyc caches
    stdin, stdout, stderr = ssh.exec_command(
        f"find {MACE_SITE}/tools/__pycache__ {MACE_SITE}/cli/__pycache__"
        f" -name '*.pyc' -delete && echo 'pyc cleared'"
    )
    print(stdout.read().decode().strip())

    sftp.close()
    ssh.close()
    print("Done.")


if __name__ == "__main__":
    deploy()
