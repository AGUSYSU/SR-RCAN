from src.model import rcan, enhance_rcan

class RCAN(rcan.RCAN):
    def __init__(self, args):
        super(RCAN, self).__init__(args)

class Enhance_RCAN(enhance_rcan.EnhancedRCAN):
    def __init__(self, rcan_model):
        super(Enhance_RCAN, self).__init__(rcan_model)

__all__ = ["RCAN", "Enhance_RCAN"]