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
    BertTokenizerFast,
    FlaxBertForSequenceClassification,
)

import spu.intrinsic as intrinsic
import spu.spu_pb2 as spu_pb2
import spu.utils.simulation as ppsim

copts = spu_pb2.CompilerOptions()
# enable x / broadcast(y) -> x * broadcast(1/y) which accelerate the softmax
copts.enable_optimize_denominator_with_broadcast = True

parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-m", "--model_path", default="examples/pretrained/bert_acc70.8_epoch9.pth", help="Path to .pth model")
parser.add_argument("-t", "--tokenizer", default="bert-base-uncased", help="Tokenizer name")
args = parser.parse_args()

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
    print(f"Running on CPU ...")
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
    print(f"CPU runtime: {(end - start)}s\noutput logits: {logits}")
    # 预测类别
    predictions = jax.nn.softmax(logits, axis=-1)
    predicted_class = jax.numpy.argmax(predictions, axis=-1)
    print(f"Predicted class: {predicted_class}, True label: {labels}")
    return predicted_class


def run_on_spu(model, input_ids, attention_masks, token_type_ids, labels):
    print(f"Running on SPU (simulation mode) ...")
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
    
    print(f"SPU runtime: {(end - start)}s\noutput logits: {logits_spu}")
    
    # 预测类别
    predictions = jax.nn.softmax(logits_spu, axis=-1)
    predicted_class = jax.numpy.argmax(predictions, axis=-1)
    print(f"SPU Predicted class: {predicted_class}, True label: {labels}")
    return predicted_class


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
        print(f"Loading PyTorch model from {checkpoint_path} and converting to Flax...")
        model = convert_pytorch_to_flax(checkpoint_path)
    
    return model


def main():
    # 加载RTE数据集
    print("Loading RTE dataset...")
    dataset_name = args.dataset
    if dataset_name == "rte":
        dataset = load_dataset("glue", "rte", split="validation")
    elif dataset_name == "boolq":
        data_root = './dataset/boolq/'
        os.makedirs(data_root, exist_ok=True)
        dataset = load_dataset('boolq', cache_dir=data_root)
    
    # 加载tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer)
    
    # 加载模型
    model = load_flax_model_from_checkpoint(args.model_path)
    
    # 统计正确率
    cpu_correct = 0
    spu_correct = 0
    total = 0
    
    for i, example in enumerate(dataset):
        if i >= 10:  # 只测试前10个样本
            break
            
        sentence1 = example["sentence1"]
        sentence2 = example["sentence2"]
        label = example["label"]
        
        # Tokenize两个句子
        encoded = tokenizer(
            sentence1,
            sentence2,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="jax"
        )
        
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        token_type_ids = encoded.get("token_type_ids", None)
        
        print(f"\n--- Example {i+1} ---")
        print(f"Sentence1: {sentence1}")
        print(f"Sentence2: {sentence2}")
        print(f"True label: {label}")
        
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
    
    # 输出总体性能
    print(f"\n=== Overall Performance ===")
    print(f"Total examples: {total}")
    print(f"CPU accuracy: {cpu_correct/total*100:.2f}%")
    print(f"SPU accuracy: {spu_correct/total*100:.2f}%")


if __name__ == "__main__":
    main()