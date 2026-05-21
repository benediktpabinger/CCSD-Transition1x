import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect("slid.fysik.dtu.dk", username="s242862", password="Butterbrot9797")

sftp = ssh.open_sftp()
sftp.putfo(__import__("io").BytesIO(b"""
from mace.tools import torch_geometric as tg
print("tg attrs:", [x for x in dir(tg) if not x.startswith('_')])
print("tg.data attrs:", [x for x in dir(tg.data) if not x.startswith('_')])
from mace.data import AtomicData
import mace.data
print("mace.data attrs:", [x for x in dir(mace.data) if not x.startswith('_')])
"""), "/tmp/chk_tg.py")
sftp.close()

_, out, _ = ssh.exec_command(
    "bash -c 'source /etc/profile && module load Python/3.13.5-GCCcore-14.3.0 && python3 /tmp/chk_tg.py 2>&1'"
)
print(out.read().decode())
ssh.close()
