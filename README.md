## Build

### 1. Prerequisite
To create an environment, run:

```
#. $HOME/anaconda3/etc/profile.d/conda.sh

conda env create -f environment.yml
conda activate bumblebee_test
```

### 2. Modify Some Python Codes
If using conda, the packages files are in "/bumblebee_test/lib/python3.10/site-packages/".

```python
# transformers/models/gpt2/modeling_flax_gpt2.py#297
def __call__(self, hidden_states, deterministic: bool = True):
    hidden_states = self.c_fc(hidden_states)
    # hidden_states = self.act(hidden_states)  
    hidden_states = jax.nn.gelu(hidden_states)
    hidden_states = self.c_proj(hidden_states)
    hidden_states = self.dropout(hidden_states, deterministic=deterministic)
    return hidden_states

# transformers/models/bert/modeling_flax_bert.py#468
def __call__(self, hidden_states):
    hidden_states = self.dense(hidden_states)
    #hidden_states = self.activation(hidden_states)
    hidden_states = jax.nn.gelu(hidden_states)
    return hidden_states

# transformers/models/vit/modeling_flax_vit.py#282
def __call__(self, hidden_states):
    hidden_states = self.dense(hidden_states)
    #hidden_states = self.activation(hidden_states)
    hidden_states = jax.nn.gelu(hidden_states)
    return hidden_states
```

```python
#jax/_src/lax/lax.py#3800
  def reducer_fn(op_val_index, acc_val_index):
    op_val, op_index = op_val_index
    acc_val, acc_index = acc_val_index
    pick_op_val = self._value_comparator(op_val, acc_val)
    return (select(pick_op_val, op_val, acc_val),
            select(pick_op_val, op_index, acc_index))
    # Pick op_val if Lt (for argmin) or if NaN
    # pick_op_val = bitwise_or(value_comparator(op_val, acc_val),
    #                          ne(op_val, op_val))
    # # If x and y are not NaN and x = y, then pick the first
    # pick_op_index = bitwise_or(pick_op_val,
    #                            bitwise_and(eq(op_val, acc_val),
    #                                        lt(op_index, acc_index)))
    # return (select(pick_op_val, op_val, acc_val),
    #         select(pick_op_index, op_index, acc_index))
```

### 3. Build Run Microbenchmarks

```sh
bazel build -c opt examples/python/ml/microbench/gelu
bazel run -c opt examples/python/microbench:gelu
```

### 4. Build and Run the Main Program
Download model weights (.pth format) and put them in "./examples/pretrained".

Run: 

```sh
bazel build -c opt examples/python/ml/flax_bert_dataset/...
bazel run -c opt examples/python/ml/flax_bert_dataset -- --model_path `pwd`/examples/pretrained/xxx.pth

```


## This Rep is copy from "BumbleBee: Secure Two-party Inference Framework for Large Transformers" [paper](https://eprint.iacr.org/2023/1678) and their repos: [BumbleBee](https://github.com/AntCPLab/OpenBumbleBee). We fix some package version problem for test.


```tex
@inproceedings{DBLP:conf/ndss/LuHGLLRHWC25,
  author       = {Wen{-}jie Lu and
                  Zhicong Huang and
                  Zhen Gu and
                  Jingyu Li and
                  Jian Liu and
                  Cheng Hong and
                  Kui Ren and
                  Tao Wei and
                  Wenguang Chen},
  title        = {{BumbleBee: Secure Two-party Inference Framework for Large Transformers}},
  booktitle    = {32nd Annual Network and Distributed System Security Symposium, {NDSS} 2025},
  publisher    = {The Internet Society},
  year         = {February 23-28, 2025},
}
```
