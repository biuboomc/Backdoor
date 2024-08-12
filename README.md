如果网站资源加载不完全，可能需要连接vpn，如果连接不上huggingface，请尝试关掉vpn使用镜像站。

最好使用16G以上显存的显卡运行，否则可能会导致结果不正确（在优化过程中可能会随机减少batch）

运行app.py即可

环境需求：
timm 1.0.3
torch 1.13.1+cu116
tqdm 4.65.0
numpy 1.25.1
matplotlib 3.7.2
pillow 10.0.0
flask 3.0.3