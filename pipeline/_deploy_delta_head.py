"""Upload delta head scripts to cluster and submit the training job."""
import paramiko

LOCAL = r"c:\Transition 1X\Transition 1x\Transition1x\pipeline\delta"
REMOTE = "/home/energy/s242862/pipeline/delta"

FILES = [
    "prepare_delta_data.py",
    "train_delta_head.py",
    "job_train_delta_head.sh",
]

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect("slid.fysik.dtu.dk", username="s242862", password="Butterbrot9797")

sftp = ssh.open_sftp()

# Ensure remote dir exists
ssh.exec_command(f"mkdir -p {REMOTE}")

for fname in FILES:
    local_path = f"{LOCAL}\\{fname}"
    remote_path = f"{REMOTE}/{fname}"
    sftp.put(local_path, remote_path)
    print(f"Uploaded {fname}")

sftp.close()

# Make shell script executable
ssh.exec_command(f"chmod +x {REMOTE}/job_train_delta_head.sh")

# Submit the job
_, out, err = ssh.exec_command(f"cd {REMOTE} && sbatch job_train_delta_head.sh")
stdout = out.read().decode().strip()
stderr = err.read().decode().strip()
print(f"\nsbatch output: {stdout}")
if stderr:
    print(f"stderr: {stderr}")

ssh.close()
