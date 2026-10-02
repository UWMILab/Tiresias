import torch


class GeneralLoRALoader:
    def __init__(self, device="cpu", torch_dtype=torch.float32):
        self.device = device
        self.torch_dtype = torch_dtype

    @staticmethod
    def _pairs(state_dict):
        pairs = {}
        for key in state_dict:
            if ".lora_up." in key:
                up_name, down_name = "lora_up", "lora_down"
            else:
                up_name, down_name = "lora_B", "lora_A"
            if up_name not in key:
                continue
            parts = key.split(".")
            position = parts.index(up_name)
            if len(parts) > position + 2:
                parts.pop(position + 1)
            parts.pop(position)
            if parts[0] == "diffusion_model":
                parts.pop(0)
            parts.pop()
            pairs[".".join(parts)] = (key, key.replace(up_name, down_name))
        return pairs

    def fuse_lora_to_base_model(self, model, state_dict, alpha=1.0):
        pairs = self._pairs(state_dict)
        modules = dict(model.named_modules())
        for name, (up_key, down_key) in pairs.items():
            module = modules[name]
            up = state_dict[up_key].to(self.device, self.torch_dtype)
            down = state_dict[down_key].to(self.device, self.torch_dtype)
            update = alpha * torch.mm(up, down)
            with torch.no_grad():
                module.weight.add_(update.to(module.weight))
        print(f"Loaded LoRA into {len(pairs)} layers.")
