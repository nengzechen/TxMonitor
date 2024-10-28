import time
from pymongo import MongoClient
import pickle
import sys
import csv
# from ITR_tree.data_process import process_entry
# from ITR_tree.build_ITR_tree import build_ITR_tree
# from ITR_tree.tokenize_text import tokenize_tree
# from ITR_tree.utilities import process_record

sys.path.append('C:\zk\LLaMA-Factory\ITR_tree\data_process.py')
import data_process
sys.path.append(r"C:\zk\LLaMA-Factory\ITR_tree\build_ITR_tree.py")
import build_ITR_tree
sys.path.append(r"C:\zk\LLaMA-Factory\ITR_tree\tokenize_text.py")
import tokenize_text
sys.path.append(r"C:\zk\LLaMA-Factory\ITR_tree\utilities.py")
import utilities

def input2token(tx_hash):
    # 连接数据库，读取数据，
    client = MongoClient('')
    dbtest = client["geth"]
    collection = dbtest.get_collection("transaction")

    query = {'tx_hash':tx_hash}
    data = collection.find(query)
    #data_onehot = collection.find()
    with open(r'C:\zk\blockGPT\vocabulary\blockgpt_finetune_val2_change2.pkl', "rb") as file:   
        vocabulary = pickle.load(file)
    
    # 处理交易：将交易三部分转换为token，并将token转换为他们在词汇表中的索引，存储在“one_hot”列表中
    # one_hot = []
    token_list = []
    #gas=0
    for idx,entry in enumerate(data):
        #gas=entry['tx_gas']
        entry = utilities.process_record(entry)
        Seqsstate_1, Seqslog_1, Seqscall_1 = data_process.process_entry(entry)
        root_node = build_ITR_tree.build_ITR_tree(Seqsstate_1, Seqslog_1, Seqscall_1)
        tokenize_text.tokenize_tree(root_node)
        for token in root_node.data:
            if vocabulary.get_index(token)!=0:
                token_list.append(token)
            else:
                # 修改unk
                token_list.append(token)
            #one_hot.append(tokenizer.convert_tokens_to_ids(token))

        while root_node.children:
            # 再判断根结点下面有没有call
            if root_node.children[0][0].tag == 'call':
                for i in  range(len(root_node.children)):
                    if i ==0:
                        continue
                    else:
                        for token in root_node.children[i][0].data:
                            if vocabulary.get_index(token)!=0:
                                token_list.append(token)
                            else:
                                token_list.append(token)
                            #one_hot.append(tokenizer.convert_tokens_to_ids(token))
                root_node = root_node.children[0][0]
            else:
                for i in range(len(root_node.children)):
                    for token in root_node.children[i][0].data:
                        if vocabulary.get_index(token)!=0:
                            token_list.append(token)
                        else:
                            token_list.append(token)
                            #one_hot.append(tokenizer.convert_tokens_to_ids(token))
                break
    return token_list

def main():
    num=0
    sum=0
    sum_gas=0
    with open(r'C:\zk\txmonitor\dataset1\normal.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            
            token_id,gas = input2token(row[0])
            token_id_len = len(token_id)
            num+=1
            sum+=token_id_len
            sum_gas+=gas
            print(num)
    print(sum/num)
    print(sum_gas/num)
if __name__ == "__main__":
    main()