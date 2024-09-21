from src.data import dataloader, srdata
from torch.utils.data.dataloader import default_collate

class Data:

    def __init__(self, args):
        kwargs = {}
        kwargs['collate_fn'] = default_collate
        kwargs['pin_memory'] = False if args.cpu else True

        self.loader_train = None

        if not args.test_only:
            trainset = srdata.SRData(args, train=True)
            self.loader_train = dataloader.MSDataLoader(args,
                                             trainset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             **kwargs)

        testset = srdata.SRData(args, train=False)

        self.loader_test = dataloader.MSDataLoader(args,
                                         testset,
                                         batch_size=1,
                                         shuffle=False,
                                         **kwargs)

