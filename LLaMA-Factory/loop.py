import yaml
import re
import subprocess

# 修改yaml文件的函数
def modify_yaml(yaml_file, model,adpater,tmp):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)

    # 修改adapter_name_or_path的值
    config['model_name_or_path'] = model
    config['template'] = tmp
    config['output_dir'] = adpater
    # 写回yaml文件
    with open(yaml_file, 'w') as file:
        yaml.dump(config, file)


# 修改yaml文件的函数
def modify_yaml1(yaml_file, model,adpater,tmp):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)

    # 修改adapter_name_or_path的值
    config['model_name_or_path'] = model
    config['template'] = tmp
    config['adapter_name_or_path'] = adpater
    # 写回yaml文件
    with open(yaml_file, 'w') as file:
        yaml.dump(config, file)

# 修改Python文件的函数，只修改第二个with open语句
def modify_py(py_file, new_file_path):
    with open(py_file, 'r') as file:
        lines = file.readlines()

    # 修改第158行内容为新的 with open 语句
    lines[157] = f"            with open(r'{new_file_path}', 'a', newline='', encoding='utf-8') as file2:\n"  # 第158行在索引中是157

    # 将修改后的内容写回文件
    with open(py_file, 'w') as file:
        file.writelines(lines)

# 执行命令的函数
def execute_command():
    command = ['llamafactory-cli', 'train', 'examples/train_lora/llama3_lora_sft.yaml']
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)

# 执行命令的函数
def execute_command1():
    command = ['llamafactory-cli', 'chat', 'examples/inference/llama3_lora_sft.yaml']
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)


# 主函数
def main():
    yaml_file = r'C:\zk\txmonitor\LLaMA-Factory\examples\train_lora\llama3_lora_sft.yaml'
    yaml_file1 = r'C:\zk\txmonitor\LLaMA-Factory\examples\inference\llama3_lora_sft.yaml'
    py_file = r'C:\zk\txmonitor\LLaMA-Factory\src\llamafactory\chat\chat_model.py'  # 你的Python文件路径

    do = ['deepseek-llm-7b-base','falcon-7b','Falcon11B','Meta-Llama-3-8B','Meta-Llama-3.1-8B','Qwen-7B','Qwen1.5-7B','Qwen2-7B','Qwen2.5-7B','Yi-6B','Yi-9B','Baichuan2-7B-Base','Baichuan2-13B-Base']
    tmp = ['deepseek','falcon','falcon','llama3','llama3','qwen','qwen','qwen','qwen','yi','yi','baichuan2','baichuan2']
    # 定义需要修改的内容，可以是一个循环或动态生成的
    for i in range(27):

        model = f'C:\\zk\\LLM_Library\\{do[i+3]}'  
        adpater = f'C:\\zk\\txmonitor\\LLaMA-Factory\\saves\\{do[i+3]}\\dora\\sft'  
        tmp1 = tmp[i+3]
        
        ####################
        # 修改YAML文件
        #modify_yaml(yaml_file, model,adpater,tmp1)
        # # 执行命令  微调训练
        # execute_command()
        # print(f'{do[i]} train OK!')

        ########
        new_file_path = f'C:\\zk\\txmonitor\\result\\dora\\{do[i+3]}.csv'  # 替换成你想要的路径

        modify_yaml1(yaml_file1, model,adpater,tmp1)

        #修改Python文件中的第二个with open语句的文件路径
        modify_py(py_file, new_file_path)


        #执行命令  微调推理
        execute_command1()
        print(f'{do[i+3]} inference OK!')

if __name__ == '__main__':
    main()