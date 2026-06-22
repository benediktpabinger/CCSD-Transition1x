import paramiko
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('slid.fysik.dtu.dk', username='s242862', key_filename=r'C:\Users\PabingerBenedikt\.ssh\dtu_key', timeout=30)
sftp = client.open_sftp()
sftp.get('/home/energy/s242862/delta_head/full_benchmark_results.json',
         r'c:\Transition 1X\Transition 1x\Transition1x\full_benchmark_results.json')
sftp.close()
client.close()
print('pulled')
