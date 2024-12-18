import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import json
import pandas as pd
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset


matplotlib.use('Agg')  # For non-interactive plots
def get_json_file_name(main_folder="data/ReaSCAN-v1.1/",split="b2"):
    if main_folder[-1] != "/":
        main_folder += "/"
    if split in ["train", "dev", "test", "dev_comp_3500"]:
        json_path = f'{main_folder}ReaSCAN-compositional/{split}.json'
    else:
        json_path = f'{main_folder}ReaSCAN-compositional-{split}/test.json'
    return json_path

def extract_percent_center(bbox, obj_size=4, fraction=0.75):
    col, row, col_plus_size, row_plus_size = bbox
    size = col_plus_size - col  # or row_plus_size - row, they should be the same
    new_size = size * fraction 
    new_size = new_size * (obj_size + 2 ) / ( 4 + 2 ) 
    offset = (size - new_size) / 2
    
    new_col = col + offset
    new_row = row + offset
    new_col_plus_size = new_col + new_size
    new_row_plus_size = new_row + new_size
    
    return [new_col, new_row, new_col_plus_size, new_row_plus_size]

def draw_object(ax, shape, color, size, row, col):
    y, x = row, col  # grid positions are inverted in matplotlib
    size_scale = size / 6  # Scale factor for size, adjust as needed
    center = (x + 0.5, y + 0.5)
    if shape == "circle":
        circle = Circle((center[0], center[1]), 0.5 * size_scale, color=color, zorder=9999)
        ax.add_patch(circle)
    elif shape == "square":
        square = Rectangle((center[0] - size_scale / 2 , center[1] - size_scale / 2), size_scale, size_scale, color=color, zorder=9999)
        ax.add_patch(square)
    elif shape == "cylinder":
        width =  0.5 * size_scale
        height =  size_scale
        # Parallelogram points, shifted and scaled based on size
        parallelogram = Polygon([
            (center[0], center[1] - height / 2),     # Upper left corner, moved right and slightly up
            (center[0] - width, center[1] + height / 2),     # Lower left corner
            (center[0] , center[1] + height / 2),     # Upper right corner, aligned with bottom right
            (center[0] + width, center[1] - height / 2)      # Lower right corner, pulling back to align with the start
        ], color=color, zorder=9999)

        ax.add_patch(parallelogram)
    elif shape == "box":
        # Assuming size specifies the number of cells on one side of the box (e.g., size=2 means a 2x2 box)
        box_size = int(size) - 0.12
        box = Rectangle((col + 0.06,  row + 0.06), box_size, box_size, fill=False, edgecolor=color, linewidth=10, zorder=9999)
        ax.add_patch(box)

def draw_all_objects(ax, situation_dict):
    object_poses = situation_dict['placed_objects'].values()    
    for obj_data in object_poses:
        row = int(obj_data['position']['row'])
        col = int(obj_data['position']['column'])
        shape = obj_data['object']['shape']
        color = obj_data['object']['color']
        size = obj_data['object']['size']
        draw_object(ax, shape, color, float(size), row, col)

def get_fig_axes():
    fig, ax = plt.subplots(figsize=(18, 18))
    ax.set_aspect(1)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.invert_yaxis() 
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.grid(True,color="black")
    ax.tick_params(tick1On=False)
    return fig, ax

def get_image(situation_dict):
    fig, ax = get_fig_axes()
    draw_all_objects(ax, situation_dict)
    ### plt to numpy
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)  # Close the figure to free memory
    del fig, ax  # Delete variables to free memory
    non_white_pixels = np.where(data.sum(axis=2) / 4 != 255)
    min_x = min(non_white_pixels[0])
    max_x = max(non_white_pixels[0])
    min_y = min(non_white_pixels[1])
    max_y = max(non_white_pixels[1])
    data = data[min_x:max_x, min_y:max_y, :3]
    return data



class ReaSCANDataset(Dataset):

    def __init__(self, split, json_path, processed_data, dataset_size=None, load_image=False, specify_objects=False, transform_images=True, load_image_from_disk=True):
        self.split = split
        self.transform_images = transform_images
        self.processed_data = processed_data
        self.load_image = load_image
        self.load_image_from_disk = load_image_from_disk
        self.load_json(json_path, dataset_size)
        if load_image:
            import jactorch.transforms.bbox as T
            self.image_transform = T.Compose([
                T.NormalizeBbox(),
                T.Resize(256),
                T.DenormalizeBbox(),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.specify_objects = specify_objects

    def load_json(self, path, size=None):
        data_list = []
        with open(path, 'r') as file:
            for index, line in enumerate(tqdm(file.readlines())):
                if size is not None and index >= size:
                    break
                data = json.loads(line)
                
                input_command = " ".join(data["input_command"])
                try:
                    program = self.processed_data[input_command][0]
                    data["program"] = program
                except KeyError:
                    data["program"] = ""
                data_list.append(data)
        
        # Convert the list of dictionaries to a DataFrame
        self.data = pd.DataFrame(data_list)
        self.data_bank = self.data.copy()
        print(f"Loaded {len(self.data)} samples from {path}")
    
    def __len__(self):
        # Assuming each entry in the JSON file is a separate data point
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve the data point at index `idx`
        data_point = self.data.iloc[idx]
        
        # Here, you can preprocess your data as needed
        input_command = data_point['input_command']
        target_sequence = data_point['target_sequence']
        target_location = np.array(data_point['target_location']).argmax()
        agent_location = torch.tensor(data_point['agent_location'], dtype=torch.float32)
        situation = torch.tensor(data_point['situation'], dtype=torch.float32)
        situation_dict = json.dumps(data_point['situation_dict'])
        program = data_point["program"]
  
        def get_pos(obj):
            return int(obj["position"]["row"]) * 6 + int(obj["position"]["column"])
        placed_objects = sorted(data_point["situation_dict"]["placed_objects"].values(), key=lambda x: get_pos(x))
        if self.specify_objects and self.load_image:
            objects = []
            locations_with_object = []

            for obj_index, obj in enumerate(placed_objects):
                row = int(obj["position"]["row"])
                col = int(obj["position"]["column"])
                if self.load_image and obj["object"]["shape"] == "box":
                    size = int(obj["object"]["size"])
                else:
                    size = 1
                bbox = [col / 6, row / 6, (col + size) / 6, (row + size) / 6]
                if obj["object"]["shape"] != "box":
                    bbox = extract_percent_center(bbox, int(obj["object"]["size"]))
                objects.append(bbox)
                locations_with_object.append(row * 6 + col)
                if obj == data_point["situation_dict"]["target_object"]:
                    target_location = obj_index
        if self.specify_objects and not self.load_image:
            objects = []
            locations_with_object = []
            for obj_index, obj in enumerate(placed_objects):
                row = int(obj["position"]["row"])
                col = int(obj["position"]["column"])
                objects.append(situation[row, col, :])
                locations_with_object.append(row * 6 + col)
                if obj == data_point["situation_dict"]["target_object"]:
                    target_location = obj_index
                
        if self.load_image:
            image, objects = self.load_images(data_point["index"], data_point["split"], objects, data_point)
        else:
            image = []
        
        num_objects = len(objects)
        locations_with_object = np.array(locations_with_object)
        if num_objects < 36:
            if self.load_image:
                objects = np.concatenate([objects, np.zeros((36 - len(objects), 4))], axis=0)
                locations_with_object = np.concatenate([locations_with_object, np.zeros(36 - len(locations_with_object))], axis=0)
            else:
                objects = np.concatenate([objects, np.zeros((36 - len(objects), situation[0,0].shape[0]))], axis=0)
                locations_with_object = np.concatenate([locations_with_object, 37 * np.ones(36 - len(locations_with_object))], axis=0)
                
        
        return {
            "index": data_point["index"],
            'input_command': " ".join(input_command),
            'image': image,
            'objects': objects,
            "locations_with_object": locations_with_object,
            'target_location': target_location,
            "num_objects": torch.tensor(num_objects),
            'agent_location': agent_location,
            'situation': situation,
            "situation_dict": situation_dict,
            "program": program
        }
    
    def filter(self, filter_func):
        self.data = self.data_bank[self.data_bank.apply(filter_func, axis=1)]
        return self

    def load_images(self, idx, split, objects, data_point):
        if self.load_image_from_disk:
            filename = f"{split}_{idx}.png"
            path = os.path.join("data/ReaSCAN-v1.1/images", filename)
            if not os.path.exists(path):
                raise Exception(f"Image not found at {path}")
            with Image.open(path) as img:
                if img.size[0] != img.size[1]:
                    print(path)
                    img = img.resize((256, 256))
                image = img.convert('RGB')
        else:
            image = get_image(data_point["situation_dict"])
            image = Image.fromarray(np.uint8(image))

        objects = np.array(objects, dtype=np.float32) * image.size[0]

        if self.transform_images:
            image, objects = self.image_transform(image, objects)
        else:
            image = np.array(image)
        
        return image, objects

    def filter_program_size_raw(self, max_length: int):
        def filt(question):
            return question['program'] is None or len(question['program']) <= max_length

        return self.filter(filt)
