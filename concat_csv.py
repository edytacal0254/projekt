import os
import pandas as pd

first_path = os.getcwd() + "\\created\\_v1\\data_v1\\data_merged_v1.csv"
second_path = os.getcwd() + "\\created\\_v2\\data_v2\\data_merged_v2.csv"
concat_path = os.getcwd() + "\\created\\_v2\\data_v2\\data_concat.csv"

if not (os.path.isfile(first_path) and os.path.isfile(second_path)):
    print("at least one of the stated files does not exist")
    exit()

first_df = pd.read_csv(first_path)
second_df = pd.read_csv(second_path)
w = ["img_name", "basic_object_name", "bg_name", "contains", "scale", "idx_c",
               "gen_x", "gen_y", "gen_w", "gen_h", "det_class", "confid",
               "det_x", "det_y", "det_w", "det_h"]
tmp1 = first_df[w]
tmp2 = second_df[w]

frames = [tmp1, tmp2]
result = pd.concat(frames)
result.to_csv(concat_path)
