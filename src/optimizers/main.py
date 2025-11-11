import torch.optim as optim
import sys

# from torch_optimizer import Shampoo
sys.path.append("src/optimizers")
import soap, adam_sania, muon, diag_hvp, weight_adamw, fat_adamw


def get_optimizer(args, model, name=None):
    # if (
    #     hasattr(args, "ft_strategy")
    #     and args.ft_strategy == "WeightLoRA"
    #     and args.optimizer not in ["weight_adamw", "fat_adamw"]
    # ):
    #     raise ValueError(
    #         "Optimizer must be 'weight_adamw' or 'fat_adamw' when using 'WeightLoRA' strategy."
    #     )
    # if (
    #     hasattr(args, "ft_strategy")
    #     and args.optimizer in ["weight_adamw", "fat_adamw"]
    #     and args.ft_strategy != "WeightLoRA"
    # ):
    #     raise ValueError(
    #         "The 'weight_adamw' or 'fat_adamw' optimizers must be used with the 'WeightLoRA' strategy."
    #     )
    optim_name = name if name else args.optimizer
    if optim_name == "adamw":
        optimizer = optim.AdamW(
            params=model.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.wd,
        )
    elif optim_name == "soap":
        optimizer = soap.SOAP(
            params=model.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            shampoo_beta=args.shampoo_beta,
            eps=args.eps,
            weight_decay=args.wd,
            precondition_frequency=args.update_freq,
        )
    elif optim_name == "sgd":
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.wd,
        )
    elif optim_name == "muon":
        optimizer = muon.Muon(
            muon_params=list(p for p in model.parameters() if p.requires_grad),
            lr=args.lr,
            adamw_betas=(args.beta1, args.beta2),
            adamw_eps=args.eps,
            adamw_wd=args.wd,
            momentum=args.momentum,
            ns_steps=args.ns_steps,
        )
    elif optim_name == "weight_adamw":
        weight_params, other_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_weight" in name:
                weight_params.append(param)
            elif "lora_A" in name or "lora_B" in name:
                other_params.append(param)
        optimizer = weight_adamw.WeightAdamW(
            [
                {"params": other_params, "name": "other_params"},
                {
                    "params": weight_params,
                    "k": args.k,
                    "proj": weight_adamw.proj_0,
                    "lr": args.lr_w,
                    "max_fat_steps": args.mfs,
                    "name": "weight_params",
                },
            ],
            lr=args.lr,
            weight_decay=args.wd,
        )
    elif optim_name == "fat_adamw":
        weight_params, loraAB_params, other_params = [], [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_weight" in name:
                weight_params.append(param)
            elif "lora_A" in name or "lora_B" in name:
                loraAB_params.append(param)
        optimizer = fat_adamw.FatAdamW(
            [
                {"params": loraAB_params, "name": "loraAB"},
                {"params": other_params, "name": "other_params"},
                {
                    "params": weight_params,
                    "proj": weight_adamw.proj_0,
                    "lr": args.lr_w,
                    "name": "weight_params",
                },
            ],
            lr=args.lr,
            weight_decay=args.wd,
            num_adapters=len(weight_params),
            fat_step=args.fat_step,
            max_fat_steps=args.mfs,
            lora_extention=args.extention,
        )
    else:
        raise NotImplementedError(f"Wrong optimizer name {args.optimizer}")
    print(optimizer)
    return optimizer
