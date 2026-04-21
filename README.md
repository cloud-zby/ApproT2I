# Approximate Cache for Text-to-Image Diffusion Models
## Installation
通过uv安装依赖
```bash
python -m venv .venv
source .venv/bin/activate

apt update
apt install -y libgl1
apt install -y libgl1-mesa-glx
apt-get update
apt-get install -y curl gnupg ca-certificates
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install -y git-lfs
git lfs install

pip install "diffusers == 0.37.0" -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install "setuptools < 75.0.0" -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install accelerate -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install faiss-cpu -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
pip install mkl -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install "transformers <= 5.4.0" -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install invisible_watermark -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install accelerate -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install safetensors -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install nvitop -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install scipy -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install scikit-image -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install git+https://github.com/openai/CLIP.git --no-build-isolation
pip install pyarrow -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pandas -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install openpyxl -i https://pypi.tuna.tsinghua.edu.cn/simple
```
## Data & Model Preparation
```bash
# 下载文本提示词数据，使得存在三个文件：dataset/COCO2014_dataset.json，dataset/DiffusionDB_dataset.parquet，dataset/PicaPic_dataset.parquet
pip install huggingface_hub -i https://pypi.tuna.tsinghua.edu.cn/simple
hf download zbyzby/ApproximateCache --local-dir ./dataset 
# 或者使用GIT LFS拉取
git lfs pull

# 下载图像数据，用于结果的评估
# MS-COCO
mkdir ./dataset/images/
mkdir ./dataset/images/MS-COCO/
wget http://images.cocodataset.org/zips/train2014.zip
unzip -n dataset/images/MS-COCO/train2014.zip -d dataset/images/MS-COCO
# DiffusionDB
mkdir ./dataset/images/DiffusionDB
export HF_ENDPOINT=https://hf-mirror.com
hf download poloclub/diffusiondb --repo-type dataset --include "images/**" --local-dir ./dataset/images/DiffusionDB
# Pick-a-Pic
mkdir ./dataset/images/Pick-a-Pic
export HF_ENDPOINT=https://hf-mirror.com
hf download pickapic-anonymous/pickapic_v1 --repo-type dataset --local-dir ./dataset/images/Pick-a-Pic

# 下载模型
modelscope download --model AI-ModelScope/stable-diffusion-2-1 --local_dir ./model_ckpt/stable-diffusion-2-1
modelscope download --model stabilityai/stable-diffusion-xl-base-1.0 --local_dir ./model_ckpt/stable-diffusion-xl-base-1.0
modelscope download --model black-forest-labs/FLUX.2-klein-base-9B --local_dir ./model_ckpt/FLUX.2-klein-base-9B
modelscope download --model google/siglip2-giant-opt-patch16-384 --local_dir ./model_ckpt/siglip2-giant-opt-patch16-384
```
## Running
You can run all-in-one script to execute the entire pipeline.
```bash
source .venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
# 以四卡模式运行
export CUDA_VISIBLE_DEVICES=0,1,2,3
# 生成10000张图片
NUM_GPUS=4 nohup ./run.sh --datasets coco,pick-a-pic,diffusiondb --step 2500 > run.log 2>&1 &
# 以两卡模式运行
export CUDA_VISIBLE_DEVICES=0,1
# 生成10000张图片
NUM_GPUS=2 nohup ./run.sh --dataset coco --step 2500 > run.log 2>&1 &
ps aux | grep '[r]un.sh'
```
Or, you can run the following commands step by step.
```bash
# 0. 设置环境变量
export HF_ENDPOINT=https://hf-mirror.com

# 1. 生成原始图像轨迹（无近似缓存）
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup python ./traj_all.py --split 0 --step 1000 > traj_0.log 2>&1 &
nohup python ./traj_all.py --split 1 --step 1000 > traj_1.log 2>&1 &
nohup python ./traj_all.py --split 2 --step 1000 > traj_2.log 2>&1 &
nohup python ./traj_all.py --split 3 --step 1000 > traj_3.log 2>&1 &
ps aux | grep '[t]raj_all.py'
cat traj.log

# 2. 生成小步数的图像轨迹（无近似缓存）
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup python ./traj_small_steps.py --split 0 --step 1000 > traj_small_steps_0.log 2>&1 &
nohup python ./traj_small_steps.py --split 1 --step 1000 > traj_small_steps_1.log 2>&1 &
nohup python ./traj_small_steps.py --split 2 --step 1000 > traj_small_steps_2.log 2>&1 &
nohup python ./traj_small_steps.py --split 3 --step 1000 > traj_small_steps_3.log 2>&1 &
ps aux | grep '[t]raj_small_steps.py'
cat traj_small_steps.log

# 3. 收集小步数图像生成过程中的注意力分数（无近似缓存）
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup python ./attention_small_steps.py --split 0 --step 1000 > attention_small_steps_0.log 2>&1 &
nohup python ./attention_small_steps.py --split 1 --step 1000 > attention_small_steps_1.log 2>&1 &
nohup python ./attention_small_steps.py --split 2 --step 1000 > attention_small_steps_2.log 2>&1 &
nohup python ./attention_small_steps.py --split 3 --step 1000 > attention_small_steps_3.log 2>&1 &
ps aux | grep '[t]raj_small_steps.py'
cat attention_small_steps.log

# 4. 检索向量数据库（文-图相似性）
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python ./retrieve.py --num_of_splits 4

# 5. 检索向量数据库（文-文相似性）
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python ./retrieve_by_prompt.py --num_of_splits 4

# 6. NIRVANA生成图像
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 nohup python ./nirvana_generate.py --num_of_splits 4 > nirvana.log 2>&1 &
ps aux | grep '[n]irvana_generate.py'
cat nirvana.log

# 7. ReCon生成图像
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 nohup python ./recon_generate.py --num_of_splits 4 > recon.log 2>&1 &
ps aux | grep '[r]econ_generate.py'
cat recon.log

# 8. ReDi生成图像
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=2 nohup python ./redi_generate.py --num_of_splits 4 > redi.log 2>&1 &
ps aux | grep '[r]edi_generate.py'
cat redi.log

# 9. MoDM生成图像
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=3 nohup python ./modm_generate.py --num_of_splits 4 > modm.log 2>&1 &
ps aux | grep '[m]odm_generate.py'
cat modm.log

# 10. 本方法生成图像（latent粒度进行融合）
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 nohup python ./ours_generate_tt.py --num_of_splits 4 > ours_tt.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python ./ours_generate_ti.py --num_of_splits 4 > ours_ti.log 2>&1 &
ps aux | grep '[o]urs_generate_tt.py'
ps aux | grep '[o]urs_generate_ti.py'
cat ours.log

# 11. 本方法生成图像（token粒度根据交叉注意力分数进行融合）
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=2 nohup python ./ours_generate_mix_tt.py --num_of_splits 4 > ours_mix_tt.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python ./ours_generate_mix_ti.py --num_of_splits 4 > ours_mix_ti.log 2>&1 &
ps aux | grep '[o]urs_generate_mix_tt.py'
ps aux | grep '[o]urs_generate_mix_ti.py'
cat ours_mix.log

# 12. 比较生成质量
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python ./compare.py
CUDA_VISIBLE_DEVICES=0 nohup python ./compare.py > compare.log 2>&1 &
```
比较结果将会保存在当前实验目录下的`comparison_images`文件夹中，包含多个SVG图和汇总的Excel表。
## Troubleshooting
如果卡住
```bash
# 4090
ssh Alex@192.168.151.232
Aiqi1115!
sudo killall -u zbyzby
Aiqi1115!
# 5090
ssh infra41@192.168.151.41
shareAI@2025
sudo killall -u zby
shareAI@2025
# Copy to H200
```bash
rsync -avzP --ignore-existing --exclude='.git' --exclude='__pycache__'  --exclude='.venv' --exclude='model_ckpt'  --exclude='dataset'  --exclude='.log' --exclude='.pkl' --exclude='.json'  /gemini-1/platform/public/zby/ApproximateT2I  zby@root@ssh-502.default@218.84.152.122://gemini/platform/public/ai_infra/zby/
```

