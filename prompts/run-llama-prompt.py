#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : run-gpt35-promptv4.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/13/2023
#
# This file is part of Project Neuro-Codex.
# Distributed under terms of the MIT license.

import re
import json
import os.path as osp
import random
# import openai
import jacinle
import jacinle.io as io
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from accelerate import PartialState
import transformers
from transformers import pipeline
import spacy
spacy_model = spacy.load("en_core_web_sm")

def setup_environment(cache_dir):
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    transformers.utils.move_cache()


class LLama2Model:
    def __init__(self, cache_dir, base_model="meta-llama/Meta-Llama-3.1-70B-Instruct") -> None:
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=base_model,
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "quantization_config": {"load_in_4bit": True},
                "cache_dir": cache_dir
                },
            # device="cuda",
        )
        self.cache = {}
        self.terminators = [
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        

    def generate_responses(self, chat, query_str):
        if query_str in self.cache:
            return self.cache[query_str]
        with torch.no_grad():
            prompt = self.pipeline.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            outputs = self.pipeline(
                prompt,
                max_new_tokens=512,
                eos_token_id=self.terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                pad_token_id=self.pipeline.tokenizer.eos_token_id
            )
        response = outputs[0]["generated_text"][len(prompt):]
        self.cache[query_str] = response
        return response

def trim_commands(command):
    command = command.replace("push","point to")
    command = command.replace("walk","point")
    command = command.replace("pull","point to")
    command = command.replace("hesitantly","")
    command = command.replace("cautiously","")
    if "while" in command:
        command = command.split("while")[0]
    return command

def get_dependency_parsing(text):
    doc = spacy_model(text)
    output = ""
    for token in doc:
        output += f"{token.text} {token.dep_} {token.head.text} {token.head.pos_} {[child for child in token.children]}\n"
    return output
    
def run_llama_2(llama_model, questions, prompts):
    questions = [trim_commands(q) for q in questions]
    query_str = '\n'.join([
        '<text> {} </text> Given the below dependency parsing {}\n'.format(q, get_dependency_parsing(q)) for q in questions
    ])
    # query_str = '\n'.join([
    #     '<text> {} </text>\n'.format(q) for q in questions
    # ])
    message = prompts['user'] + query_str + "\n"
    messages=[
                {'role': 'system', 'content': prompts['system']},
                {'role': 'user', 'content': message}
            ]
    response = llama_model.generate_responses(messages, query_str)

    return {
        'questions': questions,
        'response': response,
        'raw_response': response
    }


def fix_parentheses(string):
    # fix the parentheses matching in the string, by adding/removing brakets.

    stack = list()
    output_string = ''
    for i in range(len(string)):
        if string[i] == '(':
            stack.append(i)
            output_string += string[i]
        elif string[i] == ')':
            if len(stack) == 0:
                pass
            else:
                output_string += string[i]
                stack.pop()
        else:
            output_string += string[i]

    for i in range(len(stack)):
        output_string += ')'
    return output_string


def extract_from_gpt(results_str, expected_batch_size: int):
    # extract all programs inside the `<code></code>` tags
    results = []
    for result_str in results_str.split('<code>')[1:]:
        result_str = result_str.split('</code>')[0]
        result_str = result_str.strip()

        if result_str.startswith('describe('):
            # replace describe(xxx, iota(yyy)) with describe(xxx, lambda k: xxx(k, iota(yyy)))
            result_str = re.sub(r'describe\(([a-zA-Z]*?),\s*iota\((.*)\)\)', r'describe(\1, lambda k: \1(k, iota(\2)))', result_str)
        result_str = fix_parentheses(result_str)

        results.append(result_str)

    if len(results) != expected_batch_size:
        raise ValueError(f'Expected {expected_batch_size} results, but got {len(results)}.')
    return results

    # results = []
    # for result_line in results_str.splitlines():
    #     result_line = result_line.replace('<code>', '').replace('</code>', '').strip()
    #     if result_line.startswith('<text>') and result_line.endswith('</text>'):
    #         continue
    #     if result_line:
    #         results.append(result_line)

    # if len(questions) != len(results):
    #     print(f'Inconsistent number of questions and programs: #q={len(questions)} vs #program={len(results)}')
    #     rows = list()
    #     for q, r in itertools.zip_longest(questions, results):
    #         rows.append((q, r))
    #     print(jacinle.tabulate(rows, headers=['question', 'program']))
    # else:
    #     print(f'Found {len(questions)} questions and programs.')
    #     for question, result in zip(questions, results):
    #         mappings[question] = [result]


def main():
    parser = jacinle.JacArgumentParser()
    parser.add_argument('--dataset', type=str, default='reascan', choices=['clevr', 'referit','reascan', 'clevr-puzzle', 'clevr-rpm', 'clevr-refexp'])
    parser.add_argument('--questions', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=7.0)
    parser.add_argument('--use-user-message', action='store_true')
    parser.add_argument('--append', action='store_true')
    parser.add_argument('--based-on', type=str, default=None)
    args = parser.parse_args()

    assert args.output.endswith('.pkl')
    args.output_gpt = args.output.replace('.pkl', '.gpt.pkl')
    args.output_export = args.output.replace('.pkl', '.export.pkl')

    if args.based_on is not None:
        assert osp.exists(args.based_on)
        based_on = io.load(args.based_on)
    else:
        based_on = dict()

    if args.append:
        if not osp.exists(args.output_export):
            args.append = False
        # assert osp.exists(args.output_gpt)
        # assert osp.exists(args.output_export)
    else:
        ask = False
        if osp.exists(args.output_gpt):
            ask = True
            print(f'Output file {args.output} already exists.')
        if osp.exists(args.output_export):
            ask = True
            print(f'Output file {args.output} already exists.')
        if ask:
            if not jacinle.yes_or_no('Continue running will overwrite the existing files. Continue?', default='no'):
                return

    with open(args.prompt) as f:
        prompts_str = f.read()
        system_prmopt, user_prompt = prompts_str.split('----')
        prompts = {
            'system': system_prmopt.strip(),
            'user': user_prompt.strip()
        }

    rows = []
    rows.append(('System Prompt', prompts['system']))
    rows.append(('User Prompt', prompts['user']))
    print(args.dataset, args.dataset == 'clevr')
    if args.dataset == 'clevr':

        questions = io.load(args.questions)['questions']
        questions = sorted({q['question'] for q in questions})
    if args.dataset == 'clevr-puzzle':
        questions = io.load(args.questions)['puzzles']
        questions = sorted({q['question'] for q in questions})
    if args.dataset == 'clevr-rpm':
        questions = io.load(args.questions)['rpms']
        questions = sorted({q['question'] for q in questions})
    if args.dataset == 'clevr-refexp':
        questions = io.load(args.questions)['refexps']
        questions = sorted({q['question'] for q in questions})
    elif args.dataset == 'referit':
        import pandas as pd
        df = pd.read_csv(args.questions)
        questions = df['utterance'].tolist()
    elif args.dataset == 'reascan':
        # questions = []
        # files = ["ReaSCAN-compositional/train.json","ReaSCAN-compositional/dev.json","ReaSCAN-compositional/test.json"]
        # for i in ["a1","a2","a3","b1","b2","c1","c2"]:
        #     files.append(f"ReaSCAN-compositional-{i}/test.json")

        # # files = ["dev.json","test.json", "adverb_1.json","adverb_2.json", "contextual.json","situational_1.json","situational_2.json","visual.json","visual_easier.json","train.json"]

        # for file in files:
        #     print(f"Loading file {file}")
        #     with open(os.path.join(args.questions,file)) as f:
        #         questions += list(map(json.loads,f.readlines()))
        
        # questions = [" ".join(q["input_command"]).strip() for q in questions]
        file = args.questions
        with open(file) as f:
            questions = list(json.loads(f.readline()))
    else:
        pass
        # raise ValueError(f'Unknown dataset: {args.dataset}')

    llama_model = LLama2Model(cache_dir='./cache')
    
    if args.sample > 0:
        sampled_questions = random.sample(questions, args.sample)
        sampled_questions = list(set(sampled_questions))
    else:
        sampled_questions = list(set(questions))

    if based_on is not None:
        sampled_questions = [q for q in sampled_questions if q not in based_on]

    if not args.append:
        gpt_results = list()
        mappings = dict()
    else:
        gpt_results = io.load(args.output_gpt)
        mappings = io.load(args.output_export)

        # remove the questions that have already been processed.
        old_length = len(sampled_questions)
        sampled_questions = [q for q in sampled_questions if q not in mappings]
        print(f'Removed {old_length - len(sampled_questions)} questions that have already been processed.')
    print(len(sampled_questions))
    total_gpt_queries = 0
    meters = jacinle.GroupMeters()
    with jacinle.tqdm_pbar(total=len(sampled_questions), desc='Running LLama-3') as pbar:
        while len(sampled_questions) > 0:
            # randomly sample a batch of questions.
            questions_batch = list(random.sample(sampled_questions, min(args.batch_size, len(sampled_questions))))
            gpt_response = run_llama_2(llama_model,questions_batch, prompts)
            total_gpt_queries += 1

            results_str = gpt_response['response']
            print(results_str)
            result_batch = None
            try:
                result_batch = extract_from_gpt(results_str, args.batch_size)
            except ValueError as e:
                print(f'Error: {e}')
                

            if result_batch is not None:
                # gpt_results.append(gpt_response)
                for q, r in zip(questions_batch, result_batch):
                    mappings[q] = [r]
                sampled_questions = [q for q in sampled_questions if q not in questions_batch]
                meters.update('batch-succ', 1)
                pbar.update(len(questions_batch))
            else:
                meters.update('batch-succ', 0)

            status_values = {k: v.avg for k, v in meters.items()}
            status_values['total-gpt-queries'] = total_gpt_queries
            pbar.set_description(meters.format_simple('Runing LLama:', status_values, compressed=True))

            io.dump(args.output_gpt, gpt_results)
            io.dump(args.output_export, mappings)


if __name__ == '__main__':
    setup_environment('./cache')
    main()

