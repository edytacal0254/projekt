import os
import pandas as pd

gen_path = os.getcwd() + "\\created\\data_gen.csv"
det_path = os.getcwd() + "\\created\\data_gen.csv"
merged_path = os.getcwd() + "\\created\\data_merged.csv"

if not (os.path.isfile(gen_path) and os.path.isfile(det_path)):
    print("data_gen.csv or data_det.csv does not exist")
    exit()

if os.path.isfile(merged_path):
    os.remove(merged_path)

gen_headers = ["img_name", "basic_object_name", "bg_name", "scale", "idx_c",
               "gen_x", "gen_y", "gen_w", "gen_h"]
det_headers = ["img_name", "det_class", "confid",
               "det_x", "det_y", "det_w", "det_h"]

gen_df = pd.read_csv(gen_path, header=None, index_col=False, names=gen_headers)
det_df = pd.read_csv(det_path, header=None, index_col=False, names=det_headers)

merged_df = pd.merge(gen_df, det_df, how="outer", on="img_name")
merged_df.to_csv(merged_path)


