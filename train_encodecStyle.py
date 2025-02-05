
from collections import defaultdict
from torch.cuda.amp import GradScaler, autocast
from losses import disc_loss, total_loss
from model import EncodecModel
def train_one_step(epoch,optimizer,optimizer_disc, model, disc_model, trainloader,config,scheduler,disc_scheduler,scaler=None,scaler_disc=None,writer=None,balancer=None):
    """train one step function 

    Args:
        epoch (int): current epoch
        optimizer (_type_) : generator optimizer
        optimizer_disc (_type_): discriminator optimizer
        model (_type_): generator model
        disc_model (_type_): discriminator model
        trainloader (_type_): train dataloader
        config (_type_): hydra config file
        scheduler (_type_): adjust generate model learning rate
        disc_scheduler (_type_): adjust discriminator model learning rate
        warmup_scheduler (_type_): warmup learning rate
    """
    model.train()
    disc_model.train()
    data_length=len(trainloader)
    # Initialize variables to accumulate losses  
    accumulated_loss_g = 0.0
    accumulated_losses_g = defaultdict(float)
    accumulated_loss_w = 0.0
    accumulated_loss_disc = 0.0

    for idx,input_wav in enumerate(trainloader):
        # warmup learning rate, warmup_epoch is defined in config file,default is 5
        input_wav = input_wav.contiguous().cuda() #[B, 1, T]: eg. [2, 1, 203760]
        optimizer.zero_grad()
        with autocast(enabled=config.common.amp):
            output, loss_w, _ = model(input_wav) #output: [B, 1, T]: eg. [2, 1, 203760] | loss_w: [1] 
            logits_real, fmap_real = disc_model(input_wav)
            logits_fake, fmap_fake = disc_model(output)
            losses_g = total_loss(
                fmap_real, 
                logits_fake, 
                fmap_fake, 
                input_wav, 
                output, 
                sample_rate=config.model.sample_rate,
            ) 