# Txmonitor
a general and efficient runtime analysis framework that utilizes a fine-tuned LLM for real-time transaction detection in smart contracts
## File directory description
### Dataset
.
├── contracts   // Continue to pre-train the contract data, they relate to the transactions in our database
├── different_size_ratio    // The data scale is 500-5000, and the ratio of positive anomalies ranges from 1:1 to 1:10
├── diversity_vulnerability // 16 types of vulnerabilities
└── small_sample_dataset    // A fine-tuning example of the reentrant vulnerability is included here
### Geth
We modified geth to get the execution flow, Please refer to the tutorial on its use [geth](https://github.com/ethereum/go-ethereum)
### LLaMA-Factory
We use LLaMA-Factory to complete the fine tuning task, please refer to use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
### ML-methods
Four traditional classification models are used to compare TxMonitor
### Token
.
├── build_ITR_tree.py
├── build_WordEmbedding.py
├── build_vocabulary.py
├── data_process.py
├── input2token.py
├── load_data.py
├── token_split
│   ├── 4byte.json
│   ├── main.py
│   └── utilities.py
└── tokenize_text.py
Tokenized code for the obtained execution flow