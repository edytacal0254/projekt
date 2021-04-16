import os
import pandas as pd
import pandasql as ps

#gen_headers = ["img_name", "basic_object_name", "bg_name", "contains", "scale", "idx_c",
#               "gen_x", "gen_y", "gen_w", "gen_h"]
#det_headers = ["img_name", "det_class", "confid",
#               "det_x", "det_y", "det_w", "det_h"]
#

merg_path = os.getcwd() + "\\created\\_v0\\data_v0\\data_merged_v0.csv"
gen_dir = os.getcwd() + "\\created\\_v0\\generated_v0"
nr_of_img = len(os.listdir(gen_dir))
df = pd.read_csv(merg_path)

q1 = """SELECT COUNT(*) as 'x' FROM df"""
print(ps.sqldf(q1, locals()), "\n")

q2 = """SELECT scale, count(*) FROM df group by scale"""
print(ps.sqldf(q2, locals()), "\n")

q3 = """SELECT count(*) FROM df where det_class is null"""
print(ps.sqldf(q3, locals()), "\n")

q4 = """SELECT count(*), scale FROM df where confid is null group by scale"""
print(ps.sqldf(q4, locals()), "\n")

q5 = """SELECT img_name, COUNT(*) occ FROM df group by img_name having count(*)>1"""
print(ps.sqldf(q5, locals()), "\n")

q6 = """SELECT img_name, bg_name, scale, idx_c, det_class, confid from df where img_name == 'B14_003136.jpg' or img_name == 'B14_003138.jpg' or img_name == 'B14_003139.jpg'"""
print(ps.sqldf(q6, locals()), "\n")

