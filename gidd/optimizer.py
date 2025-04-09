import torch
import pkg_resources
import warnings

# 检查PyTorch版本
torch_version = pkg_resources.get_distribution("torch").version
is_torch_2_plus = int(torch_version.split('.')[0]) >= 2
has_compiler = hasattr(torch, 'compiler')

# 尝试导入 heavyball
has_heavyball = False
try:
    if is_torch_2_plus:  # 只在PyTorch 2.0+上尝试导入heavyball
        import heavyball
        has_heavyball = True
    else:
        warnings.warn(
            "当前PyTorch版本低于2.0，不会尝试导入heavyball包。"
            "将使用adam优化器替代。"
        )
except ImportError as e:
    if 'torch._dynamo' in str(e):
        warnings.warn(
            "无法导入 heavyball 包，因为当前 PyTorch 版本不支持 torch._dynamo。"
            "请升级到 PyTorch 2.0+ 或者使用 adam 优化器替代。"
        )
    else:
        warnings.warn(f"无法导入 heavyball 包: {e}")

def get_optimizer(config, trainer):
    params = trainer.parameters()
    if config.optimizer.type == "adam" or (config.optimizer.type == "psgd" and not has_heavyball):
        if config.optimizer.type == "psgd" and not has_heavyball:
            warnings.warn("未找到 heavyball 包，将使用 AdamW 优化器替代 PSGD。")
        optimizer = torch.optim.AdamW(
            params, 
            betas=tuple(config.optimizer.betas), 
            weight_decay=config.optimizer.weight_decay, 
            eps=config.optimizer.eps
        )
    elif config.optimizer.type == "psgd" and has_heavyball:
        # 禁用编译模式，确保在低版本PyTorch上运行
        heavyball.utils.compile_mode = None
        heavyball.utils.set_torch()
        optimizer = heavyball.ForeachPSGDKron(
            params, 
            beta=config.optimizer.beta, 
            weight_decay=config.optimizer.weight_decay, 
            mars=config.optimizer.mars, 
            caution=config.optimizer.caution
        )
        optimizer.promote = True
    return optimizer
