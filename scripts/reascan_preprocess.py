import json
import argparse
import sys
import re
from copy import copy, deepcopy
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Union
import random

def make_target_command_noisy(item):
    target_sequence = item["target_sequence"]
    vocab = ["walk", "turn left", "turn right", "pull", "push", "stay"]
    noisy_target_sequence = []
    
    if "while spinning" in " ".join(item["input_command"]) and np.random.rand() < .5:

        target_sequence = ",".join(item["target_sequence"])
        pattern = r"(turn left,turn left,turn left,turn left),(turn left|turn right|walk|pull|push)"
        if "turn left,turn left,turn left,turn left" in target_sequence:
            matches = re.findall(pattern, target_sequence)
            for match in matches:
                if np.random.rand() < .5:
                    target_sequence = re.sub(",".join(match),f"{match[1]}",target_sequence)
        noisy_target_sequence = target_sequence.split(",")
    
    elif "cautiously" in " ".join(item["input_command"]) and np.random.rand() < .5:
        target_sequence = ",".join(item["target_sequence"])
        pattern = r"(turn right,turn left,turn left,turn right),(turn left|turn right|walk|pull|push)"
        if "turn right,turn left,turn left,turn right" in target_sequence:
            matches = re.findall(pattern, target_sequence)
            for match in matches:
                target_sequence = re.sub(",".join(match),f",{match[1]}",target_sequence)
        noisy_target_sequence = target_sequence.split(",") 
    elif np.random.rand() < .5:
        for i in range(len(target_sequence)):
            if np.random.rand() < 0.1:
                pass ## delete
            else:
                noisy_target_sequence.append(target_sequence[i])
    else:
        for i in range(len(target_sequence)):
            if np.random.rand() < 0.1:
                noisy_target_sequence.append(vocab[np.random.randint(0, len(vocab))])
            else:
                noisy_target_sequence.append(target_sequence[i])
    return noisy_target_sequence

def generate_samples_with_noisy_target_commands(item, num_samples):
    return [make_target_command_noisy(item) for _ in range(num_samples)]

def get_src_address(args):

    if args.mode == "dependency_mask":
        src_address = '../../data-with-dep-mask'
    elif args.mode == "paranthesis":
        src_address = '../../data-with-paranthesis'
    elif args.mode == "constituency_mask":
        src_address = '../../data-with-mask'
    elif args.mode == "full_parsed":
        src_address = '../../data-full-parse'
    elif args.mode == "normal":
        src_address = '../../data'
    else:
        raise Exception("Invalid data mode: {args.mode}")
    return src_address

def get_file_paths(args, src_address):
    file_paths_reascan = [
                          f'{src_address}/ReaSCAN-v1.1/ReaSCAN-compositional/data-compositional-splits.txt', 
                          f'{src_address}/ReaSCAN-v1.1/ReaSCAN-compositional-a1/data-compositional-splits.txt',
                          f'{src_address}/ReaSCAN-v1.1/ReaSCAN-compositional-a2/data-compositional-splits.txt',
                          f'{src_address}/ReaSCAN-v1.1/ReaSCAN-compositional-a3/data-compositional-splits.txt',
                          f'{src_address}/ReaSCAN-v1.1/ReaSCAN-compositional-b1/data-compositional-splits.txt',
                          f'{src_address}/ReaSCAN-v1.1/ReaSCAN-compositional-b2/data-compositional-splits.txt',
                          f'{src_address}/ReaSCAN-v1.1/ReaSCAN-compositional-c1/data-compositional-splits.txt',
                          f'{src_address}/ReaSCAN-v1.1/ReaSCAN-compositional-c2/data-compositional-splits.txt',
                        ]

    file_paths_gscan = [f'{src_address}/ReaSCAN-v1.1/gSCAN-compositional_splits/dataset.txt']

    file_paths_google = [f'{src_address}/spatial_relation_splits/dataset.txt']

    if args.dataset == 'reascan':
        file_paths = file_paths_reascan
    elif args.dataset == 'gscan':
        file_paths = file_paths_gscan
    elif args.dataset == 'google':
        file_paths = file_paths_google
    else:
        raise Exception("Invalid args dataset: {args.dataset}")

    return file_paths

def get_target_loc_vector(target_dict, grid_size):
    
    target_pos = target_dict['position']
    row = int(target_pos['row'])
    col = int(target_pos['column'])
    target_loc_vector = [[0] * 6 for i in range(grid_size)]
    target_loc_vector[row][col] = 1
    
    return target_loc_vector

def parse_sparse_situation_gscan(situation_representation: dict, grid_size: int) -> np.ndarray:
    """
    Each grid cell in a situation is fully specified by a vector:
    [_ _ _ _   _       _       _    _ _ _ _    _   _ _ _ _]
     1 2 3 4 square cylinder circle y g r b  agent E S W N
     _______  _________________________ _______ ______ _______
       size             shape            color  agent agent dir.
    :param situation_representation: data from dataset.txt at key "situation".
    :param grid_size: int determining row/column number.
    :return: grid to be parsed by computational models.
    """
    if situation_representation["target_object"]:
        num_object_attributes = len([int(bit) for bit in situation_representation["target_object"]["vector"]])
    else:
        num_object_attributes = len([int(bit) for bit in situation_representation["placed_objects"]["0"]["vector"]])
    # Object representation + agent bit + agent direction bits (see docstring).
    num_grid_channels = num_object_attributes + 1 + 4

    # Initialize the grid.
    grid = np.zeros([grid_size, grid_size, num_grid_channels], dtype=int)

    # Place the agent.
    agent_row = int(situation_representation["agent_position"]["row"])
    agent_column = int(situation_representation["agent_position"]["column"])
    agent_direction = int(situation_representation["agent_direction"])
    agent_representation = np.zeros([num_grid_channels], dtype=np.int32)
    agent_representation[-5] = 1
    agent_representation[-4 + agent_direction] = 1
    grid[agent_row, agent_column, :] = agent_representation

    # Loop over the objects in the world and place them.
    placed_position = set([])
    for placed_object in situation_representation["placed_objects"].values():
        object_vector = np.array([int(bit) for bit in placed_object["vector"]], dtype=np.int32)
        object_row = int(placed_object["position"]["row"])
        object_column = int(placed_object["position"]["column"])
        placed_position.add((object_row, object_column))
        if (object_row, object_column) not in placed_position:
            grid[object_row, object_column, :] = np.concatenate([object_vector, np.zeros([5], dtype=np.int32)])
        else:
            overlay = np.concatenate([object_vector, np.zeros([5], dtype=np.int32)])
            grid[object_row, object_column, :] += overlay # simply add it.
    return grid

def parse_sparse_situation_reascan(situation_representation: dict, grid_size: int) -> np.ndarray:
    """
    Each grid cell in a situation is fully specified by a vector:
    [_ _ _ _   _       _       _     _  _ _ _ _ _ _ _ _ _ _ _ _    _   _ _ _ _]
     1 2 3 4 circle cylinder square box r b g y 1 2 3 4 r b g y  agent E S W N
     _______  _________________________ _______ _______ _______ ______ _______
       size             shape            color  box_size box_color agent agent dir.
    :param situation_representation: data from dataset.txt at key "situation".
    :param grid_size: int determining row/column number.
    :return: grid to be parsed by computational models.
    """
    if situation_representation["target_object"]:
        num_object_attributes = len([int(bit) for bit in situation_representation["target_object"]["vector"]])
    else:
        num_object_attributes = len([int(bit) for bit in situation_representation["placed_objects"]["0"]["vector"]])
    num_box_attributes = 8
    # Object representation + agent bit + agent direction bits (see docstring).
    num_grid_channels = num_object_attributes + num_box_attributes + 1 + 4

    # Initialize the grid.
    grid = np.zeros([grid_size, grid_size, num_grid_channels], dtype=int)

    # Place the agent.
    agent_row = int(situation_representation["agent_position"]["row"])
    agent_column = int(situation_representation["agent_position"]["column"])
    agent_direction = int(situation_representation["agent_direction"])
    agent_representation = np.zeros([num_grid_channels], dtype=np.int32)
    agent_representation[-5] = 1
    agent_representation[-4 + agent_direction] = 1
    grid[agent_row, agent_column, :] = agent_representation

    # Loop over the objects in the world and place them.
    placed_position = set([])
    for placed_object in situation_representation["placed_objects"].values():
        object_vector = np.array([int(bit) for bit in placed_object["vector"]], dtype=np.int32)
        if placed_object["object"]["shape"] == "box":
            box_vec_1 = np.array([int(bit) for bit in placed_object["vector"][0:4]], dtype=np.int32)
            box_vec_2 = np.array([int(bit) for bit in placed_object["vector"][8:12]], dtype=np.int32)
            box_vector = np.concatenate([box_vec_1, box_vec_2])
            object_vector[0:4] = 0
            object_vector[8:12] = 0
        else:
            box_vector = np.zeros([8], dtype=np.int32)
        object_row = int(placed_object["position"]["row"])
        object_column = int(placed_object["position"]["column"])
        placed_position.add((object_row, object_column))
        if (object_row, object_column) not in placed_position:
            grid[object_row, object_column, :] = np.concatenate([object_vector, box_vector, np.zeros([5], dtype=np.int32)])
        else:
            overlay = np.concatenate([object_vector, box_vector, np.zeros([5], dtype=np.int32)])
            grid[object_row, object_column, :] += overlay # simply add it.
    return grid

def replace_spin(command):  
    pattern = r"(turn left,turn left,turn left,turn left),(turn left|turn right|walk|pull|push)"
    if "turn left,turn left,turn left,turn left" in command:
        # command = ",".join(reversed(command.split(",")))
        matches = re.findall(pattern, command)
        # Print the group values
        for match in matches:
            command = re.sub(",".join(match),f"spin,{match[1]}",command)
    return command

def get_input_comand(data_example, args):
    if args.dataset == 'google':
        comm = data_example["command"].split(',')
        input_command = []
        for i in comm:
            input_command+=i.split(' ')
    else:
        input_command = data_example["command"].split(',')
    if args.mode == "paranthesis":
        input_command = constituency_text_parser.parse(" ".join(input_command))
        input_command_tree_masking = None
    elif args.mode == "dependency_mask":
        input_command, input_command_tree_masking = dependency_text_parser.get_parse_tree_masking(input_command)
    elif args.mode == "constituency_mask":
        input_command, input_command_tree_masking = constituency_text_parser.get_parse_tree_masking(input_command)
    elif args.mode == "full_parsed":
        input_command = constituency_text_parser.get_full_parse(" ".join(input_command))
        input_command_tree_masking = None
    else:
        input_command, input_command_tree_masking = input_command, None
    return input_command, input_command_tree_masking

def get_target_command(data_example, args):
    if args.dataset == 'gscan' and args.spin == 1 and ("spinning" in input_command or 'while spinning' in input_command):
        target_command = replace_spin(data_example["target_commands"]).split(",")
    else:
        target_command = data_example["target_commands"].split(',')
    return target_command     

def get_situation_parser(args):
    parse_sparse_situation_func = None
    if args.dataset in ['gscan', 'google']:
        parse_sparse_situation_func = parse_sparse_situation_gscan
    elif args.dataset == 'reascan':
        if args.embedding == 'modified':
            parse_sparse_situation_func = parse_sparse_situation_reascan
        elif args.embedding == 'default':
            parse_sparse_situation_func = parse_sparse_situation_gscan
    return parse_sparse_situation_func

def get_situation_representation(situation, grid_size, args):
    parse_sparse_situation_func = get_situation_parser(args)
    situation = parse_sparse_situation_func(situation, grid_size=grid_size)                
    return situation

def get_steps_data(data_example, args):
    data = {}
    parse_sparse_situation_func = get_situation_parser(args)
    steps, agent_positions, target_positions, agent_dirs = world_generator.execute(data_example, parse_sparse_situation_func)
    data["steps"] = np.array(steps, dtype=np.int32).tolist()
    data["agent_positions"] = np.array(agent_positions, dtype=np.int32).tolist()
    data["target_positions"] = np.array(target_positions, dtype=np.int32).tolist()
    data["agent_dirs"] = np.array(agent_dirs, dtype=np.int32).tolist()
    data["situation_dict"] = data_example["situation"]
    return data
             
def augment_data_for_spin(item_data):
    new_data = []
    num_samples = np.random.randint(1, 5)
    commands = random.choices(["v", "w", "x", "y", "z"], k=num_samples)

    for command in commands:
        new_item_data = deepcopy(item_data)
        new_item_data["input_command"] = ",".join(new_item_data["input_command"]).replace("push", command).split(",")
        new_item_data["target_sequence"] =  ",".join(new_item_data["target_sequence"]).replace("push", 2*command.upper()).split(",")
        new_data.append(new_item_data)
    return new_data

def data_loader(file_path: str, args) -> Dict[str, Union[List[str], np.ndarray]]:

    with open(file_path, 'r') as infile:
        all_data = json.load(infile)
        grid_size = int(all_data["grid_size"])
        splits = list(all_data["examples"].keys())
        loaded_data = {}
        for split in splits:
            loaded_data[split] = []
            print(split + ':')
            for i, data_example in enumerate(tqdm(all_data["examples"][split])):
                input_command, input_command_tree_masking = get_input_comand(data_example, args)
                target_command = get_target_command(data_example, args)
                target_location = get_target_loc_vector(data_example["situation"]["target_object"], grid_size)
                situation = get_situation_representation(data_example["situation"], grid_size, args)
                item_data = {
                    "index": i,
                    "input_command": input_command,
                    "target_sequence": target_command,
                    "target_location": target_location,
                    "agent_location": get_agent_location(data_example),
                    "situation": situation.tolist(),
                    "situation_dict": data_example["situation"]
                    }
                if input_command_tree_masking is not None:
                    item_data["mask"] = np.array(input_command_tree_masking, dtype=np.int32).tolist()
                if args.steps:
                    step_data = get_steps_data(data_example, args)
                    item_data.update(step_data)
                loaded_data[split].append(item_data)
                if args.num_samples > 0 and i > args.num_samples:
                    break  
            if split == "train" and args.augment_spin:
                augments = []
                for item_data in loaded_data[split]:          
                    if args.augment_spin and "push" in item_data["input_command"]:
                        augments.extend(augment_data_for_spin(item_data))
                loaded_data[split].extend(augments)
            if args.noise:
                for item_data in loaded_data[split]:
                    item_data["noisy_sequences"] = generate_samples_with_noisy_target_commands(item_data, args.noise)
            print(len(loaded_data[split]))
    return loaded_data

def get_agent_location(data_example):
    return int(data_example["situation"]["agent_position"]["row"]) * 6 + int(data_example["situation"]["agent_position"]["column"])


if __name__ == '__main__':
    from aux_data import make_aux_data, make_gscan_aux_data, make_google_aux_data
    from dataset.visualizer import WorldGenerator

    from parse import DKConsituencyParser, DKStanfordDependencyParser



    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='reascan', choices=['reascan', 'gscan', 'google'], help='Choose dataset for preprocessing')
    parser.add_argument('--mode', type=str, default='normal', choices=['dependency_mask', 'constituency_mask', 'normal', 'paranthesis', 'full_parsed'], help='Choose dataset for preprocessing')
    parser.add_argument('--augment_spin', type=int, default=0, help='Choose dataset for preprocessing')
    parser.add_argument('--embedding', type=str, default='modified', choices=['modified', 'default'], help='Which embedding to use')
    parser.add_argument('--steps', type=int, default=0, choices=[0,1], help='Whether to generate world state sequence')
    parser.add_argument('--spin', type=int, default=0, choices=[0,1], help='Whether to replace four turn lefts with [spin]')
    parser.add_argument('--noise', type=int, default=0, help='Number of noisy target command sequence to add')
    parser.add_argument('--num_samples', type=int, default=-1, help='Number of samples to generate for each split')
    args = parser.parse_args()

    if args.mode == "dependency_mask":
        dependency_text_parser = DKStanfordDependencyParser()    
    elif args.mode in ["paranthesis", "full_parsed","cosntituency_mask"]:
        constituency_text_parser = DKConsituencyParser()
    if args.steps:
        world_generator = WorldGenerator()

    src_address = get_src_address(args)
    file_paths = get_file_paths(args, src_address)
    print(src_address)

    for file_path in file_paths:
        print('Processing {} ...'.format("-".join(file_path.split('/')[3:5])))
        data = data_loader(file_path, args)
        for split, dt in data.items():
            print('Dumping {} json ...'.format(split))
            if args.dataset == 'reascan':
                if args.embedding == 'modified':
                    with open(file_path.split('data-compositional')[0] + split + '.json', 'w') as f:
                        for line in tqdm(dt):
                            f.write(json.dumps(line) + '\n')
                elif args.embedding == 'default':
                    with open(file_path.split('data-compositional')[0] + split + '_default_embedding.json', 'w') as f:
                        for line in tqdm(dt):
                            f.write(json.dumps(line) + '\n')
            elif args.dataset == 'gscan':
                with open(file_path.split('dataset')[0] + split + '.json', 'w') as f:
                    for line in tqdm(dt):
                        f.write(json.dumps(line) + '\n')
            elif args.dataset == 'google':
                with open(file_path.split('dataset')[0] + split + '.json', 'w') as f:
                    for line in tqdm(dt):
                        f.write(json.dumps(line) + '\n')
    
    if args.dataset == 'reascan':
        make_aux_data(f'{src_address}/ReaSCAN-v1.1/')
    elif args.dataset == 'gscan':
        make_gscan_aux_data(f'{src_address}/ReaSCAN-v1.1/')
    elif args.dataset == 'google':
        make_google_aux_data(f'{src_address}/spatial_relation_splits/')    


    
    