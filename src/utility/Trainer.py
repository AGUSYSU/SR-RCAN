import os
from src.utility import utils

import torch
from tqdm import tqdm


class Trainer():

    def __init__(self, args, loader, model, optimizer, loss):
        self.args = args
        
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = model.to(f"cuda:{args.gpu_id}")
        self.optimizer, self.loss = optimizer, loss

        self.logger = utils.Logger(args)
        self.timer = utils.Timer()

        if self.args.pre_train != '.':
            param = torch.load(self.args.pre_train, weights_only=True)
            self.model.load_state_dict(param)

        elif self.args.load != '.':
            param = torch.load(f"{self.args.load}/model/model_last.pth", weights_only=True)
            self.model.load_state_dict(param)

            checkpoint = torch.load(f"{self.args.load}/model/optimizer.pth")
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.logger.info = checkpoint['info']
        
        self.start_epoch = self.logger.info["Epoch"] + 1

        self.error_last = 1e8

    def prepare(self, l):
        device = torch.device('cpu' if self.args.cpu else f"cuda:{self.args.gpu_id}")
        def _prepare(tensor):
            if self.args.precision == 'half':
                tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(_l) for _l in l]

    def train(self):
        self.timer.record()
        
        self.logger.write_log(f"Train :")
        losses = 0
        total = 0

        self.model.train()

        loop = tqdm(self.loader_train,
                    total=len(self.loader_train),
                    ncols=100)

        for lr, hr, _ in loop:
            lr, hr = self.prepare([lr, hr])
            self.optimizer.zero_grad()

            sr = self.model(lr)

            loss = self.loss(sr, hr)

            loss.backward()
            self.optimizer.step()
            
            losses += loss.item() * hr.shape[0]
            total += hr.shape[0]
            loop.set_postfix({
                "Loss": f"{losses/total:.3f}",
                "Time": self.timer.interval()
            })
        
        self.logger.info['Loss'].append(losses / total)
        self.logger.draw_loss()
        self.logger.write_log(
            f"\tLoss : {losses/total:.3f}\n\tTime : {self.timer.interval()}")

    def test(self):
        self.timer.record()
        self.logger.write_log(f"Test :")

        self.model.eval()

        loop = tqdm(self.loader_test, ncols=100)
        psnrs = 0
        total = 0

        with torch.no_grad():
            for _, (lr, hr, _) in enumerate(loop):
                lr, hr = self.prepare([lr, hr])

                sr = self.model(lr)

                psnr = utils.calc_psnr(sr, hr)

                psnrs += psnr * hr.shape[0]
                total += hr.shape[0]
                loop.set_postfix({
                    "PSNR": f"{psnrs/total:.3f}",
                    "Time": self.timer.interval()
                })

        psnr_avg = psnrs / total
        
        self.logger.save_model(self.model, self.optimizer, psnr_avg > self.logger.info["PSNR"][-1])
        
        self.logger.info['PSNR'].append(psnr_avg)
        self.logger.draw_psnr()
        self.logger.write_log(
            f"\tPSNR : {psnr_avg:.3f}\n\tTime : {self.timer.interval()}")
        self.logger.write_log(
            f"\tBest : \n\t\tBest PSNR : {self.logger.info['best_psnr']} @ epoch {self.logger.info['best_epoch']}\n\n")
    
    def begin(self):
        print("Beging Training ...")

        for epoch in range(self.start_epoch, self.args.epochs + 1):
            print(f'\nEpoch [{epoch}/{self.args.epochs}]')

            lr = self.optimizer.param_groups[0]['lr']
            self.logger.write_log(f"[Epoch: {epoch} ] : Learning rate : {lr}")
            self.logger.info["Epoch"] = epoch

            self.train()
            self.test()
        
        print(f"------------------------------\n"
              f"Training complete\n"
              f"Best PSNR : {self.logger.info['Best']['best_psnr']}\n"
              f"The trained model weights are saved at: \n{self.logger.dir_model}")
        
    
