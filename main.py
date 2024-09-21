import torch

from src import data, model
from src.utility import param, Trainer, utils

args = param.args
torch.manual_seed(args.seed)

my_data = data.Data(args)
# my_model = model.RCAN(args)

# 加载原 RCAN 模型
rcan_model = model.RCAN(args)

optimizer, loss = utils.make_optimizer_and_loss()
trainer = Trainer.Trainer(args, my_data, rcan_model, optimizer, loss)

trainer.begin()
print('Finished .')
