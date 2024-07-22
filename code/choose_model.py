
from code.models.autoencoder import AE
from code.models.vae import VAE
from code.models.backbone_vae import BACKBONEVAE



def get_named_model(name):

    if name == "ae":
        return AE

    if name == "vae":
        return VAE


    if name== "backbone_vae":
        return BACKBONEVAE






    raise ValueError("Method does not exist")