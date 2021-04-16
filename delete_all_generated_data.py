import os
import shutil

created = os.getcwd() + "\\created"
gen = os.getcwd() + "\\created\\generated"
content = os.listdir(created)
for c in content:
    path = os.getcwd() + "\\created\\" + c
    if os.path.isfile(path):
        os.remove(path)
if os.path.isdir(gen):
    shutil.rmtree(gen)

print("Deleted all files from \\created ")
