import os
import json 
import copy

def make_folder(name):
    try: 
        os.mkdir(name) 
    except OSError as error: 
        print("Folder exist...")
    
def delete(name):
    try: 
        os.rmdir(name)
    except OSError as error: 
        print("Can't delete...") 

def save_json(name, json_dict):
    with open(name, "w") as outfile:
        json.dump(json_dict, outfile, indent=4, sort_keys=False)

CONFIG_PATH = "./configs/config.json"
CONFIG = json.load(open(CONFIG_PATH))

make_folder(CONFIG["SAVE_FOLDER"])

delete("./config_gen")
make_folder("./config_gen")

devices = [
    [
        "cuda:3",
        "cuda:3",
        "cuda:3"
    ],
    [
        "cuda:3",
        "cuda:3",
        "cuda:3"
    ],
    [
        "cuda:3",
        "cuda:3",
        "cuda:3"
    ]
]

configs = [
    [],
    [],
    []
]

for i in range(len(CONFIG["EPS"])):
    clone_cosine = copy.copy(CONFIG)
    clone_pgd = copy.copy(CONFIG)
    clone_cosine_combination = copy.copy(CONFIG)

    clone_cosine["EPS"] = CONFIG["EPS"][i]
    clone_pgd["EPS"] = CONFIG["EPS"][i]
    clone_cosine_combination["EPS"] = CONFIG["EPS"][i]

    clone_cosine["DEVICE"] = devices[i][0]
    clone_pgd["DEVICE"] = devices[i][1]
    clone_cosine_combination["DEVICE"] = devices[i][2]

    clone_cosine["ALPHA"] = CONFIG["ALPHA_COSINE"]
    clone_pgd["ALPHA"] = CONFIG["ALPHA_NORMAL"]
    clone_cosine_combination["ALPHA"] = CONFIG["ALPHA_COSINE_COMBINATE"]

    clone_cosine["SAVE_FOLDER"] = CONFIG["SAVE_FOLDER"] + "COSINE_" + str(CONFIG["EPS"][i]) + "/"
    clone_pgd["SAVE_FOLDER"] = CONFIG["SAVE_FOLDER"] + "PGD_" + str(CONFIG["EPS"][i]) + "/"
    clone_cosine_combination["SAVE_FOLDER"] = CONFIG["SAVE_FOLDER"] + "COSINE_COMBINATE_" + str(CONFIG["EPS"][i]) + "/"

    cosine_name = "./config_gen/" + "_cosine_" + str(i) + ".json"
    cosine_pgd_name = "./config_gen/" + "_pgd_" + str(i) + ".json"
    cosine_combinate_name = "./config_gen/" + "_cosine_combinate_" + str(i) + ".json"

    save_json(cosine_name, clone_cosine)
    save_json(cosine_pgd_name, clone_pgd)
    save_json(cosine_combinate_name, clone_cosine_combination)

    configs[i].append(cosine_name)
    configs[i].append(cosine_pgd_name)
    configs[i].append(cosine_combinate_name)

for c in configs:
    os.system("python cosine_test.py " + c[0])
    make_folder(json.load(open(c[0]))["SAVE_FOLDER"])

    os.system("python pgd_test.py " + c[1])
    make_folder(json.load(open(c[1]))["SAVE_FOLDER"])

    os.system("python cosine_combination_test.py " + c[2])
    make_folder(json.load(open(c[2]))["SAVE_FOLDER"])
