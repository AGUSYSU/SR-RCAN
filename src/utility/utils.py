import torch
import torch.optim as optim
import torch.nn.functional as F
import datetime
import os
import time
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Timer():

    def __init__(self):
        self.time = 0

    def record(self):
        self.time = time.time()

    def interval(self):
        return time.time() - self.time


class Logger():

    def __init__(self, args):
        self.args = args
        self.dir_data = args.dir_data
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        self.info = {
            "Epoch": 0,
            "Loss": [],
            "PSNR": [0],
            "best_psnr": 0,
            "best_epoch": 0,
        }

        if args.reset:
            os.system("rm -rf" + self.dir_data)

        self._make_dir()

        open_type = "w"
        self.log_file = open(self.dir_save + '/log.txt', open_type)

        with open(self.dir_info + "/config.txt", open_type) as f:
            f.write(now + "\n\n")
            for arg in vars(args):
                f.write(f"{arg}: {getattr(args, arg)} \n")
            f.write("\n")

    def save_model(self, model, optimizer, is_best=False):
        torch.save(model.state_dict(), self.dir_model + '/model_last.pth')
        torch.save(
            {
                'optimizer_state': optimizer.state_dict(),
                'info': self.info
            }, self.dir_model + '/optimizer.pth')

        if is_best:
            torch.save(model.state_dict(), self.dir_model + '/model_best.pth')
            self.info["best_psnr"] = self.info['PSNR'][-1]
            self.info["best_epoch"] = self.info['Epoch']

    def write_log(self, log):
        self.log_file.write(log + '\n')
        self.log_file.flush()

    def draw_psnr(self):
        epoch = range(1, len(self.info["PSNR"]) + 1)
        plt.plot(epoch, self.info["PSNR"])
        plt.xlabel("Epoch")
        plt.ylabel("PSNR")
        plt.title("PSNR")
        plt.legend()
        plt.grid()
        plt.savefig(f"{self.dir_result}/PSNR.png", dpi=1000)
        plt.close()

    def draw_loss(self):
        epoch = range(1, len(self.info["Loss"]) + 1)
        plt.plot(epoch, self.info["Loss"])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss")
        plt.legend()
        plt.grid()
        plt.savefig(f"{self.dir_result}/Loss.png", dpi=1000)
        plt.close()

    def _make_dir(self):
        self.dir_save = os.path.join(self.dir_data, 'experiment',
                                     self.args.save)
        self.dir_info = os.path.join(self.dir_save, "info")
        self.dir_model = os.path.join(self.dir_save, "model")
        self.dir_result = os.path.join(self.dir_save, "result")

        for i in [
                self.dir_save, self.dir_info, self.dir_model, self.dir_result
        ]:
            if not os.path.exists(i):
                os.makedirs(i)


def calc_psnr(target, output, max_value=255.0) -> float:
    mse = F.mse_loss(output, target)

    if mse.item() == 0:
        return torch.tensor(float('inf'))
    psnr = 10 * (torch.log10((max_value**2) / mse)).item()

    return psnr


def make_optimizer_and_loss(args, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())
    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args.momentum}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {'betas': (args.beta1, args.beta2), 'eps': args.epsilon}
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args.epsilon}

    kwargs['lr'] = args.lr
    kwargs['weight_decay'] = args.weight_decay

    loss_function = torch.nn.L1Loss()

    return optimizer_function(trainable, **kwargs), loss_function
