# TxMonitor

TxMonitor 是一个通用且高效的运行时分析框架，结合了微调后的大语言模型（LLM），用于智能合约的实时交易检测与分析。该项目集成了区块链数据采集、执行流追踪、Token化处理、传统机器学习对比和大模型微调等多模块，适用于区块链安全、异常检测、合约分析等多种场景。

## 目录结构说明

```
TxMonitor-main/
├── Dataset/           # 数据集相关，包含合约、交易、漏洞等多种数据
├── Geth/              # 修改后的Geth源码，用于区块链执行流追踪
├── LLaMA-Factory/     # LLaMA-Factory源码，用于大模型微调
├── ML_methods/        # 传统机器学习方法对比
├── Token/             # 执行流Token化与特征工程
├── README.md          # 项目说明文件
```

### Dataset

- `contracts/`  
  包含大量以太坊智能合约源码（.sol），用于预训练和分析。
- `different_size_ratio/`  
  不同规模（500~5000）和不同正负样本比例（1:1~1:10）的交易数据集，便于异常检测实验。
- `diversity_vulnerability/`  
  16类常见智能合约漏洞的标注数据（如重入、溢出、钓鱼等），每类为单独csv。
- `small_sample_dataset/`  
  小样本数据集，包含重入漏洞的微调样例。
- `labeled_tx.csv`  
  标注过的交易数据。
- `transaction_database_example.png`  
  交易数据库结构示意图。

### Geth

本项目基于以太坊官方Geth源码，进行了如下文件的定制化修改以实现执行流追踪：
- `txpool.go`
- `state_processor.go`
- `interpreter.go`
- `instructions.go`
- `blockchai.go`

**启动示例命令**（详见`Geth/README.md`）：
```bash
sudo nohup geth --datadir data/ --networkid 5555 --http --http.addr 0.0.0.0 --http.port 8545 ...
```
**数据存储**：追踪信息存入MongoDB（默认端口27018），包括call trace、state trace、log trace等。

### LLaMA-Factory

本项目集成了[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)用于大语言模型的微调。请参考其官方文档进行环境配置和模型训练。

### ML_methods

包含四种主流机器学习分类模型的实现（均基于scikit-learn）：
- `DT.py`：决策树
- `KNN.py`：K近邻
- `LR.py`：逻辑回归
- `SVM.py`：支持向量机

每个脚本均支持模型训练、测试和保存，便于与TxMonitor主方法进行对比实验。

### Token

执行流Token化与特征工程相关代码，包括：
- `build_ITR_tree.py`：构建执行流树结构
- `build_vocabulary.py`：词表构建
- `token_split/`：MongoDB数据处理与Token分割（含主处理脚本、工具函数、4byte函数签名库等）

## 快速开始

1. **环境准备**  
   - Python 3.8+，推荐使用虚拟环境
   - 安装依赖：`pip install -r requirements.txt`（如有）
   - MongoDB（默认端口27018）
   - Geth（需使用本项目修改版）

2. **数据准备**  
   - 合约、交易、漏洞等数据已包含在`Dataset/`目录下
   - 可根据需要扩展或替换

3. **执行流追踪**  
   - 参考`Geth/README.md`，启动Geth并采集链上数据

4. **Token化与特征工程**  
   - 运行`Token/token_split/main.py`等脚本，处理MongoDB中的原始追踪数据

5. **模型训练与对比**  
   - 运行`ML_methods/`下的各类模型脚本
   - 使用LLaMA-Factory进行大模型微调

## 相关开源协议与引用

### LICENSE

本项目采用MIT开源协议，详见根目录下LICENSE文件。
