import os

ots = [1.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

for ot in ots:
    os.system(f"python run.py --ot {ot}")