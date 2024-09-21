# # 多张图片超分, 图片放在 test/test_picture 下
# python test/experiment.py --gpu_id 0 --pre_train experiment/RCAN/model/model_best.pth

# # 单张图片超分
# python test/experiment.py --gpu_id 0 --img_path data/0970x4.png --pre_train experiment/RCAN/model/model_best.pth

# gradio
python test/experiment_gradio.py --gpu_id 0 --pre_train experiment/RCAN/model/model_best.pth