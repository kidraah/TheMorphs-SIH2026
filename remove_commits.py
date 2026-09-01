import sys

file_path = sys.argv[1]
with open(file_path, "r") as f:
    lines = f.readlines()

with open(file_path, "w") as f:
    for line in lines:
        if "ce981a2" in line or "62262ee" in line:
            f.write(line.replace("pick", "drop", 1))
        else:
            f.write(line)
