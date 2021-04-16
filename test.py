import os
import pandas as pd


merg_path = os.getcwd() + "\\created\\_v0\\data_v0\\data_merged_v0.csv"
gen_dir = os.getcwd() + "\\created\\_v0\\generated_v0"
nr = len(os.listdir(gen_dir))
df = pd.read_csv(merg_path)

index = df.index
number_of_rows = len(index)
print(number_of_rows - nr, " zbyt dużo, wykryto parę obiektów na jednym obrazku")

z= df.isnull().sum().sum()
z = z/6

print(z, " z ", number_of_rows, " nie wykryto")


