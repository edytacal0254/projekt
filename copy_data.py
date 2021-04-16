import os

###
only_data = False
###

gen_dir = os.getcwd() + "\\created\\generated"
data_gen_dir = os.getcwd() + "\\created\\data_gen.csv"


if os.path.isdir(gen_dir):
    new_sec_dir = os.getcwd() + "\\created\\"
    x = 0
    while os.path.isdir(new_sec_dir + "_v" + str(x)):
        x += 1
    suff = "_v" + str(x)
    new_sec_dir += suff
    print("Folder for secured data : \\created\\", suff)

    os.mkdir(os.getcwd() + "\\created\\" + suff)
    os.rename(gen_dir, new_sec_dir + "\\generated" + suff)

    files = []
    for f in os.listdir(os.getcwd() + "\\created"):
        if os.path.isfile(os.getcwd() + "\\created\\" + f):
            files.append(f)

    os.mkdir(new_sec_dir + "\\data" + suff)
    for f in files:
        os.rename(os.getcwd() + "\\created\\" + f, new_sec_dir + "\\data" + suff + "\\" + f[:len(f) - 4] + suff + ".csv")
    print(files, " and \\generated moved to \\" + suff)

else:
    dirs = []
    files = []
    nr = ""
    for d in os.listdir(os.getcwd() + "\\created"):
        if os.path.isdir(os.getcwd() + "\\created\\" + d):
            dirs.append(d)
        else:
            files.append(d)

    dirs.sort()
    for w in dirs[-1]:
        if w.isdigit():
            nr += w

    suff = "_v" + nr

    last_dir = os.getcwd() + "\\created\\" + suff + "\\data" + suff
    os.mkdir(last_dir)
    for f in files:
        os.rename(os.getcwd() + "\\created\\" + f, last_dir + "\\" + f[:len(f) - 4] + suff + ".csv")
    print(files, " moved to \\" + suff)
