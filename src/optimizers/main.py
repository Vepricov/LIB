import torch.optim as optim
import sys

# from torch_optimizer import Shampoo
sys.path.append("src/optimizers")
import soap, muon
import mikola_drop_soap


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
    elif args.optimizer == "mikola_drop_soap":
        optimizer = mikola_drop_soap.MIKOLA_DROP_SOAP(
            params=trainable_params,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            shampoo_beta=args.shampoo_beta,
            eps=args.eps,
            weight_decay=args.weight_decay,
            precondition_frequency=args.update_freq,
            max_precond_dim=args.max_precond_dim,
            init=args.init,
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
        )
    else:
        raise NotImplementedError(f"Wrong optimizer name {args.optimizer}")
    return optimizer
