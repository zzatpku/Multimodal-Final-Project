# conda create -n pointllm python=3.10 -y
conda activate pointllm
conda install cudatoolkit=11.8 -y
conda install nvidia/label/cuda-11.8.0::cuda -y
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install transformers==4.28.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install ninja -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install flash-attn==2.5.9post1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install numpy==1.26.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install openai==0.28
