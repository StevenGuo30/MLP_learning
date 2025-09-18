#!/bin/bash

# 安装带有特殊参数的 PyTorch 相关包
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

# 安装其余依赖
pip install -r requirements.txt


