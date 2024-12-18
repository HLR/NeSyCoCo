import re
import random
from dataclasses import dataclass
import pickle
import json
from tqdm import tqdm
import argparse
@dataclass
class QueryXProgram(object):
    full_program: str
    object_program: str

def get_op_type(op):
    if 'type' in op:
        return op['type']
    return op['function']

def transform(program):
    index_to_result = dict()
    variable_counter = 0

    for i, op in enumerate(program):
        op_type = get_op_type(op)
        if op_type == 'scene':
            variable_counter += 1
            index_to_result[i] = ('', f'x{variable_counter}')
        elif op_type in ('filter_size', 'filter_color', 'filter_material', 'filter_shape'):
            program_str, variable = index_to_result[op['inputs'][0]]
            this_program_str = f'{op["value_inputs"][0]}({variable})'
            program_str = this_program_str + ' and ' + program_str if program_str else this_program_str
            index_to_result[i] = (program_str, variable)
        elif op_type == 'unique':
            inner, variable = index_to_result[op['inputs'][0]]
            program_str = f'iota(Object, lambda {variable}: {inner})'
            index_to_result[i] = (program_str, None)
        elif op_type == 'relate':
            variable_counter += 1
            variable = f'x{variable_counter}'
            inner, _ = index_to_result[op['inputs'][0]]
            program_str = f'{op["value_inputs"][0]}({variable}, {inner})'
            index_to_result[i] = (program_str, variable)
        elif op_type in ('same_size', 'same_color', 'same_material', 'same_shape',  "similar_looking_color_as", "same_material_as", "similar_shape", "similar_size", "congruent_shape", "matching_color", "identical_material"):
            variable_counter += 1
            variable = f'x{variable_counter}'
            inner, _ = index_to_result[op['inputs'][0]]
            program_str = f'{op_type}({variable}, {inner})'
            index_to_result[i] = (program_str, variable)
        elif op_type == 'intersect' or op_type == 'union':
            e1, v1 = index_to_result[op['inputs'][1]]
            e2, v2 = index_to_result[op['inputs'][0]]

            if e1 == '':
                index_to_result[i] = (e2, v2)
            elif e2 == '':
                index_to_result[i] = (e1, v1)
            else:
                assert v1 in e1 and v2 in e2
                variable_counter += 1
                variable = f'x{variable_counter}'
                if op_type == 'intersect':
                    program_str = f'{e1.replace(v1, variable)} and {e2.replace(v2, variable)}'
                else:
                    program_str = f'({e1.replace(v1, variable)} or {e2.replace(v2, variable)})'
                index_to_result[i] = (program_str, variable)
        elif op_type in ('count', 'exist'):
            inner, variable = index_to_result[op['inputs'][0]]
            if inner == '':
                inner = f'thing({variable})'
            if op_type == 'exist':
                op_type = 'exists'
            program_str = f'{op_type}(Object, lambda {variable}: {inner})'
            index_to_result[i] = program_str
        elif op_type in ('query_shape', 'query_color', 'query_material', 'query_size'):
            metaconcept = op_type.split('_')[1]
            object_str, _ = index_to_result[op['inputs'][0]]
            program_str = f'describe({metaconcept.capitalize()}, lambda k: {metaconcept}(k, {object_str}))'
            index_to_result[i] = QueryXProgram(full_program=program_str, object_program=object_str)
        elif op_type == 'equal_integer':
            e1 = index_to_result[op['inputs'][0]]
            e2 = index_to_result[op['inputs'][1]]
            program_str = f'equal({e1}, {e2})'
            index_to_result[i] = program_str
        elif op_type in ('greater_than', 'less_than'):
            e1 = index_to_result[op['inputs'][0]]
            e2 = index_to_result[op['inputs'][1]]
            program_str = f'{op_type}({e1}, {e2})'
            index_to_result[i] = program_str
        elif op_type in ('equal_color', 'equal_material', 'equal_shape', 'equal_size', ):
            
            e1 = index_to_result[op['inputs'][0]]
            e2 = index_to_result[op['inputs'][1]]
            op_type = op_type.replace('equal_', 'same_')
            program_str = f'{op_type}({e1.object_program}, {e2.object_program})'
            index_to_result[i] = program_str
        else:
            raise ValueError(f'Unknown op type: {op_type}, {op}')

    ret = index_to_result[len(program) - 1]
    if isinstance(ret, QueryXProgram):
        ret = ret.full_program
    assert isinstance(ret, str)
    return ret


def replace_one_keyword(text, convert_dict, max_replacements=1):
    # Create a regex pattern to match all keywords in the convert_dict
    pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in convert_dict.keys()) + r')\b')
    
    # Find all matches in the text
    matches = pattern.findall(text)
    
    # If no matches found, return the original text
    if not matches:
        return text
    
    # Limit the number of replacements to max_replacements or the number of matches, whichever is smaller
    max_replacements = min(max_replacements, len(matches))

    # Perform the replacements
    for _ in range(max_replacements):
        match_to_replace = random.choice(matches)
        replacements = convert_dict[match_to_replace]
        replacement = random.choice(replacements)  # Select a random synonym from the list
        text = re.sub(r'\b' + re.escape(match_to_replace) + r'\b', replacement, text, count=1)
        matches.remove(match_to_replace)
        
    return text

train_convert_dict = {
    "cube": ["block"],
    "sphere": ["orb"],
    "large": ["big"],
    "small": ["tiny"],
    "metal": ["alloy"],
    "rubber": ["matte"],
    "red": ["crimson"],
    "blue": ["cobalt"],
    "brown": ["umber"],
    "yellow": ["gold"],
    "green": ["emerald"],
    "cyan": ["teal"],
    "left": ["to_the_left_of"],
    "front": ["in_front_of"],
    "same_color": ["similar_looking_color_as"],
    "same_material": ["same_material_as"],
    "same_shape": ["similar_shape", ],
}

validation_convert_dict = {
    "cube": ["box"],
    "sphere": ["ball"],
    "large": ["huge"],
    "small": ["little"],
    "metal": ["metallic"],
    "rubber": ["elastic"],
    "red": ["burgundy"],
    "blue": ["azure"],
    "brown": ["chocolate"],
    "yellow": ["mustard"],
    "left": ["left_of"],
    "front": ["front_of"],
    "same_color": ["matching_color"],
    "same_material": ["identical_material"],
    "same_shape": ["congruent_shape"],
}

if __name__ == "__main__":

    ### get data/clevr as argument argparse
    
    parser.add_argument("--data_dir", type=str, default="data/clevr")
    args = parser.parse_args()



    for folder, max_replacement in [
        (f"{args.data_dir}/val-syn-easy/", 1),
        (f"{args.data_dir}/val-syn-medium/", 3),
        (f"{args.data_dir}/val-syn-hard/", 100)
        ]:
        print("Processing folder:", folder)
        with open(f"{args.data_dir}/val/questions-ncprogram-gt.pkl", "rb") as f:
            programs = pickle.load(f)

        with open(f"{args.data_dir}/val/questions.json", "r") as f:
            data = json.load(f) 
            for item in tqdm(data["questions"]):            
                clevr_program_str = json.dumps(item["program"])
                new_clevr_program_str = replace_one_keyword(clevr_program_str, validation_convert_dict, max_replacement)
                parsed_clevr_program_str = json.loads(new_clevr_program_str)
                item["program"] = parsed_clevr_program_str

                new_program = transform(parsed_clevr_program_str)
                programs[item["question"]] = new_program
        print("Saving questions-ncprogram-gt...")
        with open(folder + "questions-ncprogram-gt.pkl", "wb") as f:
            pickle.dump(programs, f)
        print("Saving questions...")
        with open(folder + "questions.json", "w") as f:
            json.dump(data, f)
    

    for folder, max_replacement in [
        (f"{args.data_dir}/train-syn", 1),
        ]:
        print("Processing folder:", folder)
        with open(f"{args.data_dir}/train/questions-ncprogram-gt.pkl", "rb") as f:
            programs = pickle.load(f)

        with open(f"{args.data_dir}/train/questions.json", "r") as f:
            data = json.load(f) 
            for item in tqdm(data["questions"]):            
                clevr_program_str = json.dumps(item["program"])
                new_clevr_program_str = replace_one_keyword(clevr_program_str, train_convert_dict, max_replacement)
                parsed_clevr_program_str = json.loads(new_clevr_program_str)
                item["program"] = parsed_clevr_program_str
                new_program = transform(parsed_clevr_program_str)
                programs[item["question"]] = new_program


        print("Saving questions-ncprogram-gt...")
        with open(folder + "questions-ncprogram-gt.pkl", "wb") as f:
            pickle.dump(programs, f)
        print("Saving questions...")
        with open(folder + "questions.json", "w") as f:
            json.dump(data, f)
