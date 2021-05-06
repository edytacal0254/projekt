import os
from PIL import Image
import math
import random
import csv

# ________________________cords idx_c meaning
# 0 - centered
# 1 - top left corner
# 2 - bottom right corner
# 3 - left side
# 4 - bottom side
# 5 - random

# _____________________________ scales
#scale = [0.25, 0.125, 0.0725]
scale = [0.25, 0.1875, 0.125, 0.0625]
s_1 = True  # 100% scale, centered

# _____________________________images and their paths
created_dir = os.getcwd() + "\\created"
if not os.path.isdir(created_dir):
    os.makedirs(created_dir)

output_dir = os.getcwd() + "\\created\\generated"
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

input_dir = os.getcwd() + "\\objects"
backgrounds_dir = os.getcwd() + "\\backgrounds"

nr_of_generated = len(os.listdir(output_dir))

basic_img_list = os.listdir(input_dir)
basic_img_list_len = len(basic_img_list)

backgrounds_list = os.listdir(backgrounds_dir)

# ___________________________________________________________________ csv
data_path = os.getcwd() + "\\created\\data_gen.csv"
data_file = open(data_path, "w", encoding='utf-8', newline='')
data_writer = csv.writer(data_file)

# __________________________________________________________________________________________________
for idx_bon, basic_object_name in enumerate(basic_img_list):
    contains = ""
    first_l = basic_object_name[0]
    if first_l == "A":
        contains = "apple"
    elif first_l == "B":
        contains = "car"
    elif first_l == "C":
        contains = "dog"
    elif first_l == "D":
        contains = "chair"
    else:
        print("Error : unknown class / wrong file name")

    print("Processing [" + str(idx_bon + 1) + "/" + str(basic_img_list_len) + "]")

    basic_object = Image.open(input_dir + "\\" + basic_object_name)
    width, height = basic_object.size

    # for testing purposes
    # if idx_bon == 2:
    #    break

    # if not scaled down (actually objects are scaled down if they not fit in Full HD)
    if s_1:
        new_s = 1
        # checking if object fits in Full HD (in input objects only one dimension can be too big)
        if width > 1920:
            new_s = 1920 / width
        if height > 1080:
            new_s = 1080 / height
        # calc new dimensions and scale down________________________________
        new_width = math.floor(width * new_s)
        new_height = math.floor(height * new_s)
        half_new_width = math.floor(new_width / 2)
        half_new_height = math.floor(new_height / 2)

        scaled_object = basic_object.resize((new_width, new_height))

        # coordinates - only centered in that case_________________________
        cord_c = (960 - half_new_width, 540 - half_new_height)

        for bg_name in backgrounds_list:
            bg_img = Image.open(backgrounds_dir + "\\" + bg_name)

            tmp = basic_object_name.partition('.')
            new_name = tmp[0] + "_" + "{0:06}".format(nr_of_generated) + ".jpg"

            # paste object to bg, convert to jpeg save____________________________
            new_gen_img = bg_img.copy()
            new_gen_img.paste(scaled_object, cord_c, scaled_object)
            converted_img = new_gen_img.convert("RGB")
            converted_img.save(output_dir + "\\" + new_name, "jpeg")
            # csv data___________________________________________________________
            # data_row = [new_name, basic_object_name, bg_name, 100, 0]   # 100 czy new_s?
            data_row = [new_name, basic_object_name, bg_name, contains, 1, 0, cord_c[0], cord_c[1], new_width, new_height]
            data_writer.writerow(data_row)

            nr_of_generated += 1
    # end of: if s_1 (unique case) _________________________________________________________

    for s in scale:
        # calc new dimensions and scale down________________________________
        new_width = math.floor(width * s)
        new_height = math.floor(height * s)
        half_new_width = math.floor(new_width/2)
        half_new_height = math.floor(new_height/2)

        scaled_object = basic_object.resize((new_width, new_height))

        # coordinates____________________________________________
        cord_c = (960 - half_new_width, 540 - half_new_height)
        cord_tl = (0, 0)
        cord_br = ((1920-new_width), (1080-new_height))
        cord_l = (0, 540 - half_new_height)
        cord_b = (960 - half_new_width, (1080 - new_height)) #???

        random.seed(nr_of_generated)
        rand_x = random.randint(half_new_width, 1920 - half_new_width)
        rand_y = random.randint(half_new_height, 1080 - half_new_height)
        cord_rand = (rand_x - half_new_width, rand_y - half_new_height)

        cords = (cord_c, cord_tl, cord_br, cord_l, cord_b, cord_rand)

        for bg_name in backgrounds_list:
            bg_img = Image.open(backgrounds_dir + "\\" + bg_name)

            for idx_c, c in enumerate(cords):
                tmp = basic_object_name.partition('.')
                new_name = tmp[0] + "_" + "{0:06}".format(nr_of_generated) + ".jpg"
                # paste object to bg, convert to jpeg save____________________________
                new_gen_img = bg_img.copy()
                new_gen_img.paste(scaled_object, c, scaled_object)
                converted_img = new_gen_img.convert("RGB")
                converted_img.save(output_dir + "\\" + new_name, "jpeg")

                # added column, object location and size
                data_row = [new_name, basic_object_name, bg_name, contains, s, idx_c, c[0], c[1], new_width, new_height]
                data_writer.writerow(data_row)

                nr_of_generated += 1

data_file.close()
print("Generated " + str(len(os.listdir(output_dir))) + " images")
