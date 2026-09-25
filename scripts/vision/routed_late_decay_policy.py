"""Predeclared epoch-wise late decay; no validation-dependent decisions."""
VERSION='routed-late-linear-decay-48epochs-v1'
BASE_LR=0.00025

def factor(epoch):
    if not isinstance(epoch,int) or isinstance(epoch,bool) or not 0<=epoch<48:
        raise ValueError('Expected one of 48 frozen epochs')
    # First 240 steps unchanged; epochs 24..47 descend to half baseline LR.
    return 1.0 if epoch<24 else 1.0-0.5*(epoch-23)/24

def sequence():
    return [BASE_LR*factor(step//10) for step in range(480)]
