# Robust watermarking method for embedding logo



## Dependencies and Installation
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux)).
- [tensorflow = 1.1.14](https://www.tensorflow.org/) .
- Download dependencies

~~~bash
# 1. 克隆环境
conda env create -f TF114.yaml

# 2. 安装依赖
pip install -r requirements.txt
~~~




## Get Started
- Run `python main.py` for training.

~~~bash
# Recommended training methods for embedding logo image
python main.py --exp_name watermark_logo --cover_h 224 --cover_w 224 --num_epochs 500 --batch_size 4 --lr .00001 --dataset_path /home/Dataset/train/mirflickr --cover_mse_ratio 2 --cover_lpips_ratio 1.2 --wm_mse_ratio 8 --wm_lpips_ratio 1 --GPU 0 --only_secret_N 5200 --gauss_stddev 0.01 --max_warp 0.01 --max_bri 0.01 --rnd_sat 0.01 --max_hue 0.01 --cts_low 0.9 --cts_high 1.1 --mse_gain 100 --mse_gain_epoch 100
~~~



## Dataset
- In this paper, we use the commonly used dataset Mirflickr, and ImageNet.

- For train on your own dataset, change the code in `main.py`:

    `line23:  --dataset_path = 'your dataset' ` 



##  Tensorboard

~~~bash
cd code path
tensorboard --logdir ./logs --port 6006
~~~

