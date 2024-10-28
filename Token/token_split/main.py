from utilities import process_record
import pymongo
import multiprocessing
import time
from multiprocessing import Semaphore
import sys
import json

def process_output_handle(queue_in, queue_out):
    # handle BSONObjectTooLarge error
    cnt = 0
    while True:
        item = queue_in.get()
        if item == "STOP":
            print(f"process_output_handle:{cnt}")
            queue_out.put("STOP")
            break
        cnt+=1
        if sys.getsizeof(json.dumps(item)) > 15 * 1024 * 1024:  # 15MB as a buffer
            print(f"Warning: Large document of size {sys.getsizeof(json.dumps(item))} bytes: {item['tx_hash']}")
        else:
            queue_out.put({'tx_hash': item['tx_hash'],'call':item['call']})


def process_database_writer(queue, batch_size=100):
    MongoClient_out = pymongo.MongoClient(host="", port=27018,username="",password="")
    
    ##### 需要存放的数据库 ######
    collection_out = MongoClient_out['geth']['test_attack_processed']
    cnt = 0
    batch_data = []  # 用于保存批量数据的列表
    while True:
        data = queue.get()
        if data == "STOP":
            # 如果存在未插入的数据，插入它们
            print(f"process_database_writer:{cnt}")
            if batch_data:
                collection_out.insert_many(batch_data)
                batch_data.clear()
            break
        cnt+=1
        batch_data.append(data)

        # 当达到批量大小时进行写入
        if len(batch_data) == batch_size:
            collection_out.insert_many(batch_data)
            batch_data.clear()  # 清空列表以准备下一批数据
    MongoClient_out.close()


if __name__ == "__main__":
    # 初始化消息队列
    queue_process = multiprocessing.Manager().Queue(500)
    queue_db = multiprocessing.Manager().Queue(500)
    
    # 创建独立的返回数据处理进程
    output_handle_processes = multiprocessing.Process(target=process_output_handle, args=(queue_process, queue_db,))
    output_handle_processes.start()
    
    # 创建独立的数据库写入进程
    db_writer_process = multiprocessing.Process(target=process_database_writer, args=(queue_db,))
    db_writer_process.start()

    # 主进程
    # 远程连接服务器中mongodb，选中transaction集合
    MongoClient = pymongo.MongoClient(host="10.12.46.33", port=27018,username="b515",password="sqwUiJGHYQTikv6z")

    #####  需要修改  #####
    collection = MongoClient['geth']['test_attack']
    # 查询规则
    # query = {
    #     "tx_blocknum": {"$gt": 5000000, "$lt": 6000000},
    #     "tx_trace": {"$ne": ""}
    # }
    query = {}
    cursor = collection.find(query).batch_size(2000)
    st = time.time()
    cnt = 0
    for item in cursor:
        cnt+=1
        process_record(item, queue_process)
    print(f"all:{cnt}")
    queue_process.put("STOP")

    # 等待其它进程完成
    output_handle_processes.join()
    db_writer_process.join()
    print(time.time()-st)