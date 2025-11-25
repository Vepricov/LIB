import torch.optim as optim
import sys

# from torch_optimizer import Shampoo
sys.path.append("src/optimizers")
import soap, muon
import lora_rite
import adamuon, rmsspectral, adam_sania


def get_optimizer(args, model):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "adamw":
        optimizer = optim.AdamW(
            params=trainable_params,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "soap":
        optimizer = soap.SOAP(
            params=trainable_params,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            shampoo_beta=args.shampoo_beta,
            eps=args.eps,
            weight_decay=args.weight_decay,
            precondition_frequency=args.update_freq,
            max_precond_dim=args.max_precond_dim,
        )
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(
            params=trainable_params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "muon":
        optimizer = muon.Muon(
            muon_params=trainable_params,
            lr=args.lr,
            adamw_betas=(args.beta1, args.beta2),
            adamw_eps=args.eps,
            adamw_wd=args.weight_decay,
            momentum=args.momentum,
            ns_steps=args.ns_steps,
            orth_algo=args.orth_algo,
        )
    elif args.optimizer == "adamuon":
        optimizer = adamuon.AdaMuon(
            params=trainable_params,
            lr=args.lr,
            momentum=args.momentum,
            ns_steps=args.ns_steps,
            weight_decay=args.weight_decay,
            adamw_lr=args.adamw_lr,
            adamw_betas=(args.beta1, args.beta2),
            adamw_eps=args.eps,
            adamw_wd=args.weight_decay,
            eps=args.eps,
        )
    elif args.optimizer == "rmsspectral":
        optimizer = rmsspectral.RMSSpectral(
            params=trainable_params,
            lr=args.lr,
            momentum=args.momentum,
            ns_steps=args.ns_steps,
            weight_decay=args.weight_decay,
            adamw_lr=args.adamw_lr,
            adamw_betas=(args.beta1, args.beta2),
            adamw_eps=args.eps,
            adamw_wd=args.weight_decay,
            eps=args.eps,
            rms_power=getattr(args, "rms_power", 0.25),
        )
    elif args.optimizer == "rmsspectral_sania":
        optimizer = rmsspectral.RMSSpectral(
            params=trainable_params,
            lr=args.lr,
            momentum=args.momentum,
            ns_steps=args.ns_steps,
            weight_decay=args.weight_decay,
            adamw_lr=args.adamw_lr,
            adamw_betas=(args.beta1, args.beta2),
            adamw_eps=args.eps,
            adamw_wd=args.weight_decay,
            eps=args.eps,
            rms_power=0.5,
        )
    elif args.optimizer == "lora_rite":
        optimizer = lora_rite.LoRARite(
            params=trainable_params,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adam_sania":
        optimizer = adam_sania.AdamSania(
            params=trainable_params,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.weight_decay,
        )
    else:
        raise NotImplementedError(f"Wrong optimizer name {args.optimizer}")
    return optimizer
