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
