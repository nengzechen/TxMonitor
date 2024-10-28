import logging
import sys
import os
from dataclasses import dataclass
from typing import Dict, Sequence, Union, List
import datasets
import torch
from datasets import load_dataset, concatenate_datasets
import transformers

sys.path.append('C:\zk\LLaMA-Factory\ITR_tree\input2token.py')
import input2token

logger = logging.getLogger('__name__')
IGNORE_INDEX = -100
# 用于增加知识
# PROMPT_TEMPLATE = (
#         "Below is a question for introducing opcode-related content."
#         "You need to study the command questions and answers carefully to understand the opcode sequence.\n\n"
#         "### Question:\n{instruction}\n\n{input}### Answer:\n{output}</s>"
#     )
# 用于指令
PROMPT_TEMPLATE = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}</s>"
    )
def build_instruction_dataset(data_path: Union[List[str],str],
                tokenizer: transformers.PreTrainedTokenizer,
                max_seq_length: int, data_cache_dir = None,
                preprocessing_num_workers = None,
                ):

    def tokenization(examples):
        sources = []
        targets = []
        # 增加内容
        all_texts = []
        prompt = PROMPT_TEMPLATE
        for instruction, input, output in zip(examples['instruction'],examples['input'],examples['output']):
            
            input_itr_int_list = input2token.input2token(input)
            input_itr_int_list = input_itr_int_list[:1024]
            # 将整数ID列表转换会他们对应的文本token，上一步的逆操作
            # input_itr_str_list = tokenizer.convert_ids_to_tokens(input_itr_int_list)
            # 将转换回的文本token列表拼接成一个完整的字符串，
            input_itr_str = ' '.join(input_itr_int_list)
            #   instruction = instruction + '\n' + input_itr_str
            source = prompt.format_map({'instruction':instruction,'input':input_itr_str,'output':output})
            source =   '<s>'+source 
            all_texts.append(source)
            target = f"{output}{tokenizer.eos_token}"

            sources.append(source)
            targets.append(target)

        tokenized_sources = tokenizer(sources,return_attention_mask=False)
        tokenized_targets = tokenizer(targets,return_attention_mask=False,add_special_tokens=False)

        all_input_ids = []
        all_labels = []
        # 增加text
        for s,t in zip(tokenized_sources['input_ids'],tokenized_targets['input_ids']):
            # 
            input_ids = torch.LongTensor(s+t)[:max_seq_length]
            labels = torch.LongTensor([IGNORE_INDEX] * len(s) + t)[:max_seq_length]
            # 增加text内容
            text = tokenizer.decode(input_ids)
            # assert len(input_ids) == len(labels)
            all_input_ids.append(input_ids)
            all_labels.append(labels)


        results = {'input_ids':all_input_ids, 'labels': all_labels}
        # results = {'text':all_texts}
        return results


    logging.warning("building dataset...")
    all_datasets = []

    if not isinstance(data_path,(list,tuple)):
        data_path = [data_path]
    for file in data_path:
        if data_cache_dir is None:
            data_cache_dir = str(os.path.dirname(file))
        cache_path = os.path.join(data_cache_dir,os.path.basename(file).split('.')[0])
        os.makedirs(cache_path, exist_ok=True)
        try:
            processed_dataset = datasets.load_from_disk(cache_path)
            logger.info(f'training datasets-{file} has been loaded from disk')
        except Exception:
            raw_dataset = load_dataset("json", data_files=file, cache_dir=cache_path)
            tokenization_func = tokenization
            tokenized_dataset = raw_dataset.map(
                tokenization_func,
                batched=True,
                num_proc=preprocessing_num_workers,
                remove_columns=["instruction","input","output"],
                keep_in_memory=False,
                desc="preprocessing on dataset",
            )
            processed_dataset = tokenized_dataset
            processed_dataset.save_to_disk(cache_path)
        processed_dataset.set_format('torch')
        all_datasets.append(processed_dataset['train'])
    all_datasets = concatenate_datasets(all_datasets)
    print("zk-all_datasets:", all_datasets)
    return all_datasets