import subprocess

commad = ['python3', 'save_txt_video.py']
with open('log_1.txt', 'w')as f:
    subprocess.run(commad, stdout=f, stderr=subprocess.STDOUT)

