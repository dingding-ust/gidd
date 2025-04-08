import torch
import heavyball
import pkg_resources

# 检查PyTorch版本
torch_version = pkg_resources.get_distribution("torch").version
is_torch_2_plus = int(torch_version.split('.')[0]) >= 2
has_compiler = hasattr(torch, 'compiler')

def get_optimizer(config, trainer):
    params = trainer.parameters()
    if config.optimizer.type == "adam":
        optimizer = torch.optim.AdamW(params, betas=tuple(config.optimizer.betas), weight_decay=config.optimizer.weight_decay, eps=config.optimizer.eps)
    elif config.optimizer.type == "psgd":
        # 禁用编译模式，确保在低版本PyTorch上运行
        heavyball.utils.compile_mode = None
        heavyball.utils.set_torch()
        optimizer = heavyball.ForeachPSGDKron(params, beta=config.optimizer.beta, weight_decay=config.optimizer.weight_decay, mars=config.optimizer.mars, caution=config.optimizer.caution)
        optimizer.promote = True
        # heavyball.utils.fused_hook(params, heavyball.ForeachPSGDKron, beta=config.optimizer.beta, mars=config.optimizer.mars)
    return optimizer
