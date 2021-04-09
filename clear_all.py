import os

data_file = open(os.getcwd() + "\\data.csv", "w+")
data_file.close()

output_dir = os.getcwd() + "\\generated"
for f in os.listdir(output_dir):
    os.remove(output_dir + "\\" + f)

print("Cleared generated content")
