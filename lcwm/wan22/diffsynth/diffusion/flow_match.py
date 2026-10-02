import torch


class FlowMatchScheduler:
    def __init__(self):
        self.training = False
        self.sigmas = torch.empty(0)
        self.timesteps = torch.empty(0)
        self.linear_timesteps_weights = torch.empty(0)

    def set_timesteps(
        self,
        num_inference_steps: int = 100,
        denoising_strength: float = 1.0,
        shift: float = 5.0,
        training: bool = False,
    ) -> None:
        sigma_start = float(denoising_strength)
        sigmas = torch.linspace(sigma_start, 0.0, num_inference_steps + 1)[:-1]
        self.sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        self.timesteps = self.sigmas * 1000
        self.training = training
        if training:
            values = torch.exp(-2 * ((self.timesteps - 500) / 1000) ** 2)
            values = values - values.min()
            self.linear_timesteps_weights = values * (1000 / values.sum())

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        index = self._index(timestep)
        sigma = self.sigmas[index]
        next_sigma = 0 if index + 1 >= len(self.sigmas) else self.sigmas[index + 1]
        return sample + model_output * (next_sigma - sigma)

    def add_noise(
        self,
        original: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        sigma = self.sigmas[self._index(timestep)]
        return (1 - sigma) * original + sigma * noise

    @staticmethod
    def training_target(
        sample: torch.Tensor,
        noise: torch.Tensor,
        _timestep: torch.Tensor,
    ) -> torch.Tensor:
        return noise - sample

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        return self.linear_timesteps_weights[self._index(timestep)]

    def _index(self, timestep: torch.Tensor) -> torch.Tensor:
        return torch.argmin((self.timesteps - timestep.detach().cpu()).abs())
