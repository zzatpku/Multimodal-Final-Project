conda create -n pointllm python=3.10 -y
conda activate pointllm
conda install cudatoolkit=11.8 -y
conda install nvidia/label/cuda-11.8.0::cuda -y
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install transformers==4.28.0
pip install -e .
pip install ninja
pip install flash-attn==2.5.9post1
pip install opencv-python
pip install numpy==1.26.4
pip install tokenizers==0.13.3
pip install protobuf==3.20.0
