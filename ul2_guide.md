<!--
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->

# UL2 Inference on GPU

UL2 model is new LLM model that is quite similar to T5 but trained with a different objective and scaling knobs. UL2 was trained using Jax and T5X infrastructure. In this guide, we provide a solution to convert the UL2 model to NVIDIA FasterTransformer format and serve with NVIDIA Triton Inference container. 

The FasterTransformer UL2/T5 implementation are in [t5_guide.md](https://github.com/NVIDIA/FasterTransformer/blob/main/docs/t5_guide.md). 

## Table Of Contents
 
- [UL2 Inference on GPU](#ul2-inference-on-gpu)
  - [Table Of Contents](#table-of-contents)
  - [Setup Environment](#setup-environment)
    - [NVIDIA NGC Containers](#nvidia-ngc-containers)
    - [Build Conversion Container](#build-conversion-container)
  - [Model Conversion](#model-conversion)
    - [Run JAX UL2 Converter](#run-jax-ul2-converter)
    - [Model Testing and Benchmarking](#model-testing-and-benchmarking)
  - [Run Serving on Single Node](#run-serving-on-single-node)
    - [How to set the model configuration](#how-to-set-the-model-configuration)
    - [Prepare Triton T5 model store](#prepare-triton-t5-model-store)
    - [Run serving directly](#run-serving-directly)

## Setup Environment

To set up UL2 inference environment on GPU, we will build 2 docker environments, 1 for conversion and 1 for serving. 

### NVIDIA NGC Containers

The NVIDIA NGC catalog hosts containers for the top AI and data science software, tuned, tested and optimized by NVIDIA. We use NGC containers as base to build conversion container and pull bigNLP EA container as the serving container. 

NVIDIA GPU container runtime requires installation of [NVIDIA docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html), GCP DLVM images has NVIDIA docker preinstalled.

### Build Conversion Container

First, clone NVIDIA [FasterTransformer project](https://github.com/NVIDIA/FasterTransformer), and copy fine-tuned UL2 checkpoint to a local directory, for example `/home/ul2/models`.

We build docker container and build the FasterTransformer project inside the container, we specify SM version as `80` for A100 GPUs, user can change to `70` if they opt to use V100. 
```
docker run -ti --gpus all --shm-size 5g -p 9999:9999 -v $PWD:$PWD -v /home/ul2/models:/models -w $PWD --name ft-converter nvcr.io/nvidia/pytorch:22.07-py3 bash

cd FasterTransformer
mkdir build
cd build
cmake -DSM=80 -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON ..
make -j12
pip install -r ../examples/pytorch/t5/requirement.txt
pip install transformers==4.20.1
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Model Conversion

### Run JAX UL2 Converter

First we copy `jax_t5_ckpt_convert.py` and `ul2_config.template` to `examples/pytorch/t5/utils/`. In `FasterTransformer/build` directory, we run following command  insdie conversion container to convert the UL2 JAX checkpoint 

```
python3 ../examples/pytorch/t5/utils/jax_t5_ckpt_convert.py /models/ul2-xsum /models/ul2_ft --tensor-parallelism 2
```

### Model Testing and Benchmarking

To test the FT model, we can run following scripts
```
mpirun -n 2 python3 ../examples/pytorch/t5/summarization.py --ft_model_location /models/ul2_ft --hf_model_location /models/ul2-xsum --test_ft --data_type bf16 --cache_path /tmp/cache --tensor_para_size 2
```

## Run Serving on Single Node

Per the configuration during conversion, we will use 4xA100 to run Triton inference server to serve the converted model, The inference server we use is the EA release of NVIDIA bigNLP inference container, 

### How to set the model configuration

In UL2 FasterTransformer Triton backend, the serving configuration is controlled by `config.pbtxt`. We provide an example in `triton/config.pbtxt`. It contains the input parameters, output parameters, some other settings like `tensor_para_size` and `model_checkpoint_path`. 

We use the `config.ini` in the `model_checkpoint_path` to control the model hyper-parameters like head number, head size and transformer layers. The `config.ini` will generated by checkpoint converter when user convert model. User can also change the setting to run tests on custom model size to benchmark.  

The following table shows the details of these settings:

* Settings in config.pbtxt

| Classification |             Name             |              Tensor/Parameter Shape              | Data Type |                                                                 Description                                                                 |
| :------------: | :--------------------------: | :----------------------------------------------: | :-------: | :-----------------------------------------------------------------------------------------------------------------------------------------: |
|     input      |                              |                                                  |           |                                                                                                                                             |
|                |         `input_ids`          |          [batch_size, max_input_length]          |  uint32   |                                                        input ids after tokenization                                                         |
|                |      `sequence_length`       |                   [batch_size]                   |  uint32   |                                                     real sequence length of each input                                                      |
|                |       `runtime_top_k`        |                   [batch_size]                   |  uint32   |                                                 **Optional**. candidate number for sampling                                                 |
|                |       `runtime_top_p`        |                   [batch_size]                   |   float   |                                               **Optional**. candidate threshold for sampling                                                |
|                | `beam_search_diversity_rate` |                   [batch_size]                   |   float   |                     **Optional**. diversity rate for beam search in this [paper](https://arxiv.org/pdf/1611.08562.pdf)                      |
|                |        `temperature`         |                   [batch_size]                   |   float   |                                                     **Optional**. temperature for logit                                                     |
|                |        `len_penalty`         |                   [batch_size]                   |   float   |                                                   **Optional**. length penalty for logit                                                    |
|                |     `repetition_penalty`     |                   [batch_size]                   |   float   |                                                 **Optional**. repetition penalty for logit                                                  |
|                |        `random_seed`         |                   [batch_size]                   |  uint64   |                                                   **Optional**. random seed for sampling                                                    |
|                |    `is_return_log_probs`     |                   [batch_size]                   |   bool    |                                    **Optional**. flag to return the log probs of generated token or not.                                    |
|                |       `max_output_len`       |                   [batch_size]                   |  uint32   |                                                  **Optional**. max output sequence length                                                   |
|                |         `beam_width`         |                   [batch_size]                   |  uint32   |                                   **Optional**. beam size for beam search, using sampling if setting to 1                                   |
|                |       `bad_words_list`       |          [batch_size, 2, word_list_len]          |   int32   |  **Optional**. List of tokens (words) to never sample. Should be generated with FasterTransformer/examples/pytorch/gpt/utils/word_list.py   |
|                |      `stop_words_list`       |          [batch_size, 2, word_list_len]          |   int32   | **Optional**. List of tokens (words) that stop sampling. Should be generated with FasterTransformer/examples/pytorch/gpt/utils/word_list.py |
|     output     |                              |                                                  |           |                                                                                                                                             |
|                |         `output_ids`         |           [batch_size, beam_width, -1]           |  uint32   |                                                      output ids before detokenization                                                       |
|                |      `sequence_length`       |                   [batch_size]                   |  uint32   |                                                     real sequence length of each output                                                     |
|                |       `cum_log_probs`        |             [batch_size, beam_width]             |   float   |                                         **Optional**. cumulative log probability of output sentence                                         |
|                |      `output_log_probs`      | [batch_size, beam_width, request_output_seq_len] |   float   |                              **Optional**. It records the log probability of logits at each step for sampling.                              |
|   parameter    |                              |                                                  |           |                                                                                                                                             |
|                |      `tensor_para_size`      |                                                  |    int    |                                                   parallelism ways in tensor parallelism                                                    |
|                |     `pipeline_para_size`     |                                                  |    int    |                                                  parallelism ways in pipeline parallelism                                                   |
|                |         `data_type`          |                                                  |  string   |                                     infernce data type: fp32 = float32, fp16 = float16, bf16 = bfloat16                                     |
|                |  `enable_custom_all_reduce`  |                                                  |   bool    |                                                       use custom all reduction or not                                                       |
|                |         `model_type`         |                                                  |  string   |                                                                must use `T5`                                                                |
|                |   `model_checkpoint_path`    |                                                  |  string   |                                             the path to save `config.ini` and weights of model                                              |

### Prepare Triton T5 model store

Following the guide in FasterTransformer backend to prepare the docker image, or to use the EA release of bigNLP inference container. 

We put the converted model into the following structure, with provided config.pbtxt:
```
  triton_model
    ├──ul2 
        ├── 1
        │   ├── 2-gpu
        │   └── config.ini 
        └── config.pbtxt                    
```

### Run serving directly

First start the inference container with NGC bigNLP EA inference container 
```
docker run -ti --gpus '"device=0,1,2,3"' --shm-size 5g --ulimit memlock=-1 --ulimit stack=67108864 \
       -v $PWD:$PWD -v /home/ul2/models:/models -w $PWD --network host --security-opt seccomp=unconfined \
       --name ul2-infer nvcr.io/ea-bignlp/bignlp-inference:22.08-py3
```

Then starting the triton server
```
tritonserver --model-repository=/models/triton_model
```

Before sending inference request, set up the preprocessing dependencies for the client. And modify [summarization.py](https://github.com/triton-inference-server/fastertransformer_backend/blob/main/tools/t5_utils/summarization.py) to change the model name to ul2 per our model structure. Run `summarization.py` to send inference request to Triton server. 
```
pip install -r ../examples/pytorch/t5/requirement.txt
python3 /fastertransformer_backend/tools/t5_utils/summarization.py --ft_model_location /models/ul2_ft/2-gpu --hf_model_location /models/ul2  --test_ft --data_type=bf16 --cache_path /tmp/cache
```

The results would be like
```bash
 Summary :  A bumbling sheriff who chased the moonshine-running Duke boys back and forth across the back roads of a fictitious Georgia county has died.</s>.
```

