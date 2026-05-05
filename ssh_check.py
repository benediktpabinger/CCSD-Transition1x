import paramiko
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('slid.fysik.dtu.dk', username='s242862', password='Butterbrot9797')
stdin, stdout, stderr = client.exec_command('squeue -u s242862 | grep mace && echo "---" && tail -8 /home/energy/s242862/logs/mace_train_10193298.log')
print(stdout.read().decode())
client.close()
