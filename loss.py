from losses import *
# from config import PARAMS_CONFIG
# from utils import get_params


loss_dict = {
    'cka': CKALoss,
    'consine': CosineSimilarityLoss(),
    'absconsine': AbsCosineSimilarityLoss(),
    'dcl': DCLLoss(temperature=0.07),
    'directclr': directCLR(),
    'hypersphere': HypersphereLoss(),
    'infonce': InfoNCE(),
    'rincev1': RINCE(),
    'rincev2': RINCEV2(),
    'simclrv1': SimCLRv1(),
    'simclrv2': SimCLRv2(),
    'tico': TiCo(),
}

def get_loss(loss_name):
    # loss_name = get_params(PARAMS_CONFIG)['model_params']['contrative_loss']
    return loss_dict[loss_name]