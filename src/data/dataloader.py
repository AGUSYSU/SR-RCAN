import torch
from torch.utils.data import DataLoader

class MSDataLoader(DataLoader):
    def __init__(self, args, dataset, batch_size=1, shuffle=False,
                 sampler=None, batch_sampler=None,
                 collate_fn=None, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        super(MSDataLoader, self).__init__(dataset,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           sampler=sampler,
                                           batch_sampler=batch_sampler,
                                           num_workers=args.n_threads,
                                           collate_fn=collate_fn,
                                           pin_memory=pin_memory,
                                           drop_last=drop_last,
                                           timeout=timeout,
                                           worker_init_fn=worker_init_fn)
        self.scale = args.scale

    def __iter__(self):
        return super(MSDataLoader, self).__iter__()
