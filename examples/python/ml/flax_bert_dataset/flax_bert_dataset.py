# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import time
from contextlib import contextmanager

import os
import flax.linen as fnn
import jax
import jax.nn as jnn
import numpy as np
import torch
from datasets import load_dataset
from flax import serialization
from flax.training import checkpoints
from transformers import (
    AutoTokenizer,
    BertConfig,
    BertTokenizer,
    FlaxBertForSequenceClassification,
)

import spu.intrinsic as intrinsic
import spu.spu_pb2 as spu_pb2
import spu.utils.simulation as ppsim

import datetime
import sys
import logging



copts = spu_pb2.CompilerOptions()
# enable x / broadcast(y) -> x * broadcast(1/y) which accelerate the softmax
copts.enable_optimize_denominator_with_broadcast = True

parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-m", "--model_path", default="/examples/pretrained/bert_acc72.9_epoch14_boolqpure.pth", help="Path to .pth model")
parser.add_argument("-t", "--tokenizer", default="bert-base-uncased", help="Tokenizer name")
parser.add_argument("-l", "--log_path", default="examples/logs/", help="Path to log file")

args = parser.parse_args()

# 获取脚本所在的目录

# 配置日志
# os.makedirs(args.log_path, exist_ok=True)
current_file_path = os.path.abspath(__file__)
print(f"current_file_path: {current_file_path}")
# 获取当前文件所在目录
current_dir = os.path.dirname(current_file_path)
print(f"current_dir: {current_dir}")
# 获取项目根目录（假设当前文件在 examples/python/ml/flax_bert_dataset/ 下）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))

print(f"project_root: {project_root}")
# 生成带日期时间戳的日志文件名
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"bert_inference_{current_time}.log"
log_path = os.path.join(project_root, log_filename)


# 配置logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 重定向print到logging
print = logger.info
logger.info(f"log create at {log_path}")

# 使用模拟模式代替分布式模式
config = spu_pb2.RuntimeConfig(
    protocol=spu_pb2.ProtocolKind.CHEETAH, field=spu_pb2.FieldType.FM64
)
# 可以根据需要添加更多配置
# config.enable_hal_profile = True
# config.experimental_enable_colocated_optimization = False
# config.cheetah_2pc_config.enable_mul_lsb_error = True
# config.cheetah_2pc_config.approx_less_precision = 4

# 创建模拟器，2表示模拟2个节点
SIM = ppsim.Simulator(2, config)


def _gelu(x):
    return intrinsic.spu_gelu(x)


def _softmax(x, axis=-1, where=None, initial=None):
    x_max = jax.numpy.max(x, axis, where=where, initial=initial, keepdims=True)
    x = x - x_max
    # spu.neg_exp will clip values that too large.
    # nexp = jax.numpy.exp(x) * (x > -14.0)
    nexp = intrinsic.spu_neg_exp(x)
    divisor = jax.numpy.sum(nexp, axis, where=where, keepdims=True)
    return nexp / divisor


@contextmanager
def hijack(enabled=True):
    if not enabled:
        yield
        return
    # hijack some target functions
    jnn_gelu = jnn.gelu
    fnn_gelu = fnn.gelu
    jnn_sm = jnn.softmax
    fnn_sm = fnn.softmax

    jnn.gelu = _gelu
    fnn.gelu = _gelu
    jnn.softmax = _softmax
    fnn.softmax = _softmax

    yield
    # recover back
    jnn.gelu = jnn_gelu
    fnn.gelu = fnn_gelu
    jnn.softmax = jnn_sm
    fnn.softmax = fnn_sm


def run_on_cpu(model, input_ids, attention_masks, token_type_ids, labels):
    # logger.info(f"Running on CPU ...")
    params = model.params

    def eval(params, input_ids, attention_masks, token_type_ids):
        logits = model(
            input_ids, 
            attention_masks, 
            token_type_ids=token_type_ids,
            params=params
        )[0]
        return logits

    start = time.time()
    logits = eval(params, input_ids, attention_masks, token_type_ids)
    end = time.time()
    # logger.info(f"CPU runtime: {(end - start)}s\noutput logits: {logits}")
    # 预测类别
    predictions = jax.nn.softmax(logits, axis=-1)
    predicted_class = jax.numpy.argmax(predictions, axis=-1)
    logger.info(f"CPU runtime: {(end - start)}s; Predicted: {predicted_class}, True label: {labels}")
    return predicted_class


def run_on_spu(model, input_ids, attention_masks, token_type_ids, labels):
    # logger.info(f"Running on SPU (simulation mode) ...")
    params = model.params

    def eval(params, input_ids, attention_masks, token_type_ids):
        with hijack(enabled=True):
            logits = model(
                input_ids, 
                attention_masks, 
                token_type_ids=token_type_ids,
                params=params
            )[0]
        return logits

    # 使用模拟模式运行SPU函数
    start = time.time()
    spu_fn = ppsim.sim_jax(SIM, eval, copts=copts)
    logits_spu = spu_fn(params, input_ids, attention_masks, token_type_ids)
    end = time.time()
    
    # logger.info(f"SPU runtime: {(end - start)}s\noutput logits: {logits_spu}")
    
    # 预测类别
    predictions = jax.nn.softmax(logits_spu, axis=-1)
    predicted_class = jax.numpy.argmax(predictions, axis=-1)
    logger.info(f"SPU runtime: {(end - start)}s; Predicted: {predicted_class}, True label: {labels}")
    return predicted_class

#这个numlabels 之后需要好好处理一下 这里统一是2
def convert_pytorch_to_flax(pytorch_model_path, num_labels=2):
    """
    将PyTorch的.pth模型转换为Flax格式
    """
    # 加载PyTorch模型
    from transformers import BertForSequenceClassification as TorchBertForSequenceClassification
    
    # 创建PyTorch模型
    torch_model = TorchBertForSequenceClassification.from_pretrained(
        args.tokenizer, 
        num_labels=num_labels
    )
    
    # 加载.pth权重
    state_dict = torch.load(pytorch_model_path, map_location=torch.device('cpu'))
    torch_model.load_state_dict(state_dict)
    
    # 创建对应的Flax模型
    config = BertConfig.from_pretrained(args.tokenizer, num_labels=num_labels)
    flax_model = FlaxBertForSequenceClassification(config=config)
    
    # 转换权重（简化版，实际可能需要更复杂的转换）
    # 这里我们使用Flax模型保存再加载的方式
    # 首先保存为临时文件
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 保存PyTorch模型为HuggingFace格式
        torch_model.save_pretrained(tmpdir)
        # 从保存的格式加载Flax模型
        flax_model = FlaxBertForSequenceClassification.from_pretrained(tmpdir, from_pt=True)
    
    return flax_model


def load_flax_model_from_checkpoint(checkpoint_path):
    """
    从Flax checkpoint加载模型
    """
    # 如果已经有转换好的Flax模型，直接加载
    try:
        model = FlaxBertForSequenceClassification.from_pretrained(checkpoint_path)
    except:
        # 否则尝试从.pth转换
        logger.info(f"Loading PyTorch model from {checkpoint_path} and converting to Flax...")
        model = convert_pytorch_to_flax(checkpoint_path)
    
    return model
def llama_preprocess_function(examples, tokenizer, max_length, dataset_name = "boolq"):
    if "sst2" in dataset_name.lower():
        texts = examples['sentence']
    elif "qnli" in dataset_name.lower():
        texts = [(q, s) for q, s in zip(examples['question'], examples['sentence'])]
    elif "rte" in dataset_name.lower():
        texts = [(s1, s2) for s1, s2 in zip(examples['sentence1'], examples['sentence2'])]
    elif "snli" in dataset_name.lower():
        # SNLI数据集：前提(premise)和假设(hypothesis)
        texts = [(p, h) for p, h in zip(examples['premise'], examples['hypothesis'])]
    elif "boolq" in dataset_name.lower():
        # BoolQ数据集：段落(passage)和问题(question)
        texts = [(p, q) for p, q in zip(examples['passage'], examples['question'])]
    elif "xnli" in dataset_name.lower():
        # XNLI数据集：前提(premise)和假设(hypothesis)
        texts = [(p, h) for p, h in zip(examples['premise'], examples['hypothesis'])]
    elif "squad" in dataset_name.lower():
        # SQuAD数据集：上下文(context)和问题(question)
        texts = [(c, q) for c, q in zip(examples['context'], examples['question'])]
    else: 
        print("llama_preprocess_function wrong dataset_name")
        exit(1)
    # Tokenize the texts
    if isinstance(texts[0], tuple):  # For sentence pair tasks
        result = tokenizer(
            [t[0] for t in texts],
            [t[1] for t in texts],
            truncation=True,
            padding=False,
            max_length=max_length,
        )
    else:  # For single sentence tasks
        result = tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=max_length,
        )
    
    # 处理不同数据集的标签字段
    if 'label' in examples:
        result['labels'] = examples['label']
    elif 'labels' in examples:  # 有些数据集用labels
        result['labels'] = examples['labels']
    elif 'answer' in examples:  # BoolQ用answer字段
        result['labels'] = [1 if ans else 0 for ans in examples['answer']]
    elif 'answers' in examples:  # SQuAD数据集
        # 对于SQuAD，我们将任务简化为判断是否有答案
        # 如果有答案文本，标签为1，否则为0
        result['labels'] = [1 if len(ans['text']) > 0 else 0 for ans in examples['answers']]
    else: 
        print("llama_preprocess_function wrong examples")
        exit(1)
    return result

def main():
    # 加载RTE数据集


    data_root = './dataset/boolq/'
    os.makedirs(data_root, exist_ok=True)
    dataset = load_dataset('boolq', cache_dir=data_root)
    logger.info(f"Loading dataset from {data_root}")
    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    
    # 加载模型
    model = load_flax_model_from_checkpoint(args.model_path)
    logger.info(f"Loading model from {args.model_path}")
    validation_dataset = dataset["validation"]
    encoded_dataset = validation_dataset.map(
        lambda examples: llama_preprocess_function(examples, tokenizer, max_length=128, dataset_name = "boolq"),
        batched=True,
        remove_columns=validation_dataset.column_names,
    )
    # 统计正确率
    cpu_correct = 0
    spu_correct = 0
    total = 0
    logger.info(f"dataset contains {len(encoded_dataset)} samples")
    for i, example in enumerate(encoded_dataset):
        # if i >= 10:  # 只测试前10个样本
        #     break
            
        input_ids = np.array(example["input_ids"]).reshape(1, -1)
        attention_mask = np.array(example["attention_mask"]).reshape(1, -1)
        token_type_ids = np.array(example.get("token_type_ids", [])).reshape(1, -1)
        label = example["labels"]
        
        # 获取原始文本用于显示
        # original_example = validation_dataset[i]
        # question = original_example["question"]
        # passage = original_example["passage"]
        # answer = original_example["answer"]
        
        logger.info(f"\n--- Example {i+1} ---")
        # logger.info(f"Question: {question}")
        # logger.info(f"Passage: {passage[:100]}..." if len(passage) > 100 else f"Passage: {passage}")
        # logger.info(f"True answer: {answer} (label: {1 if answer else 0})")
        
        # 在CPU上运行
        cpu_pred = run_on_cpu(model, input_ids, attention_mask, token_type_ids, label)
        
        # 在SPU上运行
        spu_pred = run_on_spu(model, input_ids, attention_mask, token_type_ids, label)
        
        # 统计正确率
        total += 1
        if cpu_pred == label:
            cpu_correct += 1
        if spu_pred == label:
            spu_correct += 1
        if i % 10 == 0:
    # 输出总体性能
            # logger.info(f"\n=== Overall Performance ===")
            logger.info(f"Total {total}: CPU acc: {cpu_correct/total*100:.2f}% | SPU acc: {spu_correct/total*100:.2f}%")


if __name__ == "__main__":
    main()