
import json
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
from tqdm import tqdm
from PIL import Image
matplotlib.use('Agg')  # For non-interactive plots
from reascan import  draw_object, draw_all_objects, get_fig_axes, get_image
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def get_pos_and_color(object_data):
    pos = int(object_data["position"]["row"]) * 6 + int(object_data["position"]["column"])
    return pos, object_data["object"]

def generate_and_save_image(situation_dict, split, index):
    filename = f"{split}_{index}.png"
    path = os.path.join("data/ReaSCAN/images", filename)        
    if os.path.exists(path):
        try:
            Image.open(path).verify()
            return
        except:
            pass
    print(f"Generating {path}")
    data = get_image(situation_dict)
    plt.imsave(path, data)
    plt.close()

def load_json(path, split):
    futures = []
    with open(path, 'r') as file:
        with ThreadPoolExecutor(max_workers=96) as executor:
            for index, line in enumerate(tqdm(file.readlines())):
                data = json.loads(line)
                futures.append(executor.submit(generate_and_save_image, data["situation_dict"], split, data["index"]))
                # generate_and_save_image(data["situation_dict"], split, data["index"])
    for future in as_completed(futures):
        try:
            res = future.result()  # Retrieve result (or raise exception if any)
        except Exception as e:
            print(f"An error occurred: {e}")
def load_json_serial(path, split):
    with open(path, 'r') as file:
        for index, line in enumerate(tqdm(file.readlines())):
            data = json.loads(line)
            generate_and_save_image(data["situation_dict"], split, data["index"])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str, default="all")
    parser.add_argument("--mode", type=str, default="parallel")
    args = parser.parse_args()

    total_time = 0
    main_folder = "data/ReaSCAN/"
    if args.file_name == "all":
        files = ["ReaSCAN-compositional/train.json","ReaSCAN-compositional/dev.json","ReaSCAN-compositional/test.json"]
        for i in ["a1","a2","a3","b1","b2","c1","c2"]:
            files.append(f"ReaSCAN-compositional-{i}/test.json")
    else:
        files = [args.file_name]
    for file in files:
        path = os.path.join(main_folder, file)
        print(f"Loading {path}")
        split = file.split("/")[0]
        if "ReaSCAN-compositional/" in file:
            split = file.split("/")[1].split(".")[0]
        else:
            split = file.split("/")[0].split("-")[-1]
    
        start_time =time.time()
        if args.mode == "parallel":
            load_json(path, split)
        else:
            load_json_serial(path, split)
        end_time = time.time()
        duration = end_time - start_time
        total_time += duration
        print(f"{split} Ended in {duration}")
    print(f"Total time {total_time}")