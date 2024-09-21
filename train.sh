LOG="experiment/RCAN`date +%Y-%m-%d-%H-%M-%S`.out"

# 参数解释
# save 保存文件夹名
# pretrain 加载模型路径
# load 断点续训, 保存文件夹名

nohup python main.py --save RCAN --gpu_id 0 --batch_size 32 --repeat 40 > $LOG &

nohup python main.py --save RCAN --pre_train experiment/RCAN/model/model_best.pth --gpu_id 0 --batch_size 32 --repeat 40 > $LOG &

nohup python main.py --save RCAN --load RCAN --gpu_id 0 --batch_size 32 --repeat 40 > $LOG &