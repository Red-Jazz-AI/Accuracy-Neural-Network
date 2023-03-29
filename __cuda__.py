import torch

def isCuda():
    
    isCuda_ = torch.cuda.is_available()
    if isCuda:
        return print("Cuda is avaliable")
    else:
        return ("Cuda is not avaliable")
    

def _isCuda():

    return torch.cuda.is_available()
