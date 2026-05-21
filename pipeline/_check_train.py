import paramiko
import sys

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect("slid.fysik.dtu.dk", username="s242862", password="Butterbrot9797")

_, out, _ = ssh.exec_command("squeue -u s242862 -h 2>&1")
print("=== squeue ===")
print(out.read().decode("utf-8", errors="replace"))

_, out, _ = ssh.exec_command("tail -60 /home/energy/s242862/logs/eval_delta_10381688.log 2>&1")
print("=== eval log ===")
sys.stdout.buffer.write(out.read().replace(b'\x1b', b'').replace(b'\r', b'\n'))
print()

ssh.close()
