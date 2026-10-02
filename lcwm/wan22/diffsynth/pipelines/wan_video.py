import torch
from einops import rearrange
from PIL import Image
from tqdm import tqdm

from ..configs.model_configs import WAN_TI2V_5B
from ..core.gradient.gradient_checkpoint import gradient_checkpoint_forward
from ..core.loader.config import ModelConfig
from ..core.loader.model import load_model
from ..diffusion.flow_match import FlowMatchScheduler
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit
from ..models.wan_video_dit import WanModel, sinusoidal_embedding_1d
from ..models.wan_video_text_encoder import HuggingfaceTokenizer, WanTextEncoder
from ..models.wan_video_vae import WanVideoVAE38
from ..utils.state_dict_converters.wan_video_vae import convert_wan_vae_state_dict


class WanVideoPipeline(BasePipeline):
    def __init__(
        self,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(
            device=device,
            torch_dtype=torch_dtype,
            height_division_factor=16,
            width_division_factor=16,
            time_division_factor=4,
            time_division_remainder=1,
        )
        self.scheduler = FlowMatchScheduler()
        self.tokenizer: HuggingfaceTokenizer | None = None
        self.text_encoder: WanTextEncoder | None = None
        self.dit: WanModel | None = None
        self.vae: WanVideoVAE38 | None = None
        self.units = [
            WanVideoUnitShapeChecker(),
            WanVideoUnitNoiseInitializer(),
            WanVideoUnitPromptEmbedder(),
            WanVideoUnitInputVideoEmbedder(),
            WanVideoUnitImageEmbedder(),
        ]
        self.model_fn = model_fn_wan_video

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype,
        device: str | torch.device,
        model_configs: list[ModelConfig],
        tokenizer_config: ModelConfig,
    ) -> "WanVideoPipeline":
        if len(model_configs) != 3:
            raise ValueError("expected DiT, umT5, and VAE model files")
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        pipe.dit = load_model(
            WanModel,
            model_configs[0].path,
            config=WAN_TI2V_5B,
            torch_dtype=torch_dtype,
            device=device,
        )
        pipe.text_encoder = load_model(
            WanTextEncoder,
            model_configs[1].path,
            torch_dtype=torch_dtype,
            device=device,
        )
        pipe.vae = load_model(
            WanVideoVAE38,
            model_configs[2].path,
            torch_dtype=torch_dtype,
            device=device,
            state_dict_converter=convert_wan_vae_state_dict,
        )
        pipe.height_division_factor = pipe.vae.upsampling_factor * 2
        pipe.width_division_factor = pipe.vae.upsampling_factor * 2
        pipe.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_config.path,
            seq_len=512,
            clean="whitespace",
        )
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        input_image: Image.Image | None = None,
        seed: int | None = None,
        rand_device: str = "cpu",
        height: int = 256,
        width: int = 256,
        num_frames: int = 9,
        cfg_scale: float = 5.0,
        num_inference_steps: int = 30,
        sigma_shift: float = 5.0,
        action_context: torch.Tensor | None = None,
        representation_context: torch.Tensor | None = None,
        tiled: bool = True,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
        progress_bar=tqdm,
    ):
        self.scheduler.set_timesteps(
            num_inference_steps,
            denoising_strength=1.0,
            shift=sigma_shift,
        )
        inputs_positive = {"prompt": prompt}
        inputs_negative = {"negative_prompt": negative_prompt}
        inputs_shared = {
            "input_image": input_image,
            "input_video": None,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "seed": seed,
            "rand_device": rand_device,
            "cfg_scale": cfg_scale,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
        }
        for unit in self.units:
            inputs_shared, inputs_positive, inputs_negative = self.unit_runner(
                unit, self, inputs_shared, inputs_positive, inputs_negative
            )
        if action_context is not None:
            inputs_positive["action_context"] = action_context
        if representation_context is not None:
            inputs_positive["representation_context"] = representation_context

        for step, timestep in enumerate(progress_bar(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(
                dtype=self.torch_dtype, device=self.device
            )
            positive = self.model_fn(
                dit=self.dit,
                latents=inputs_shared["latents"],
                timestep=timestep,
                context=inputs_positive["context"],
                action_context=inputs_positive.get("action_context"),
                representation_context=inputs_positive.get("representation_context"),
                fuse_vae_embedding_in_latents=inputs_shared.get(
                    "fuse_vae_embedding_in_latents", False
                ),
            )
            if cfg_scale == 1.0:
                prediction = positive
            else:
                negative = self.model_fn(
                    dit=self.dit,
                    latents=inputs_shared["latents"],
                    timestep=timestep,
                    context=inputs_negative["context"],
                    fuse_vae_embedding_in_latents=inputs_shared.get(
                        "fuse_vae_embedding_in_latents", False
                    ),
                )
                prediction = negative + cfg_scale * (positive - negative)
            inputs_shared["latents"] = self.scheduler.step(
                prediction,
                self.scheduler.timesteps[step],
                inputs_shared["latents"],
            )
            if "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, :1] = inputs_shared[
                    "first_frame_latents"
                ]

        video = self.vae.decode(
            inputs_shared["latents"],
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        video = self.vae_output_to_video(video)
        return video


class WanVideoUnitShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(("height", "width", "num_frames"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(
            height, width, num_frames
        )
        return {"height": height, "width": width, "num_frames": num_frames}


class WanVideoUnitNoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(("height", "width", "num_frames", "seed", "rand_device"))

    def process(
        self, pipe: WanVideoPipeline, height, width, num_frames, seed, rand_device
    ):
        frames = (num_frames - 1) // 4 + 1
        shape = (
            1,
            pipe.vae.model.z_dim,
            frames,
            height // pipe.vae.upsampling_factor,
            width // pipe.vae.upsampling_factor,
        )
        return {"noise": pipe.generate_noise(shape, seed=seed, rand_device=rand_device)}


class WanVideoUnitPromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            positive_inputs={"prompt": "prompt"},
            negative_inputs={"prompt": "negative_prompt"},
        )

    def process(self, pipe: WanVideoPipeline, prompt: str):
        ids, mask = pipe.tokenizer(
            prompt, return_mask=True, add_special_tokens=True
        )
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)
        lengths = mask.gt(0).sum(dim=1).long()
        context = pipe.text_encoder(ids, mask)
        for length in lengths:
            context[:, length:] = 0
        return {"context": context}


class WanVideoUnitInputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            (
                "input_video",
                "noise",
                "tiled",
                "tile_size",
                "tile_stride",
            )
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        input_video,
        noise,
        tiled,
        tile_size,
        tile_stride,
    ):
        if input_video is None:
            return {"latents": noise}
        video = pipe.preprocess_video(input_video)
        input_latents = pipe.vae.encode(
            video,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        latents = pipe.scheduler.add_noise(
            input_latents, noise, timestep=pipe.scheduler.timesteps[0]
        )
        return {"latents": latents}


class WanVideoUnitImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            (
                "input_image",
                "latents",
                "height",
                "width",
                "tiled",
                "tile_size",
                "tile_stride",
            )
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        input_image,
        latents,
        height,
        width,
        tiled,
        tile_size,
        tile_stride,
    ):
        if input_image is None:
            return {}
        image = pipe.preprocess_image(input_image.resize((width, height))).transpose(0, 1)
        first_frame = pipe.vae.encode(
            [image],
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        latents[:, :, :1] = first_frame
        return {
            "latents": latents,
            "fuse_vae_embedding_in_latents": True,
            "first_frame_latents": first_frame,
        }


def model_fn_wan_video(
    dit: WanModel,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    action_context: torch.Tensor | None = None,
    representation_context: torch.Tensor | None = None,
    fuse_vae_embedding_in_latents: bool = False,
    use_gradient_checkpointing: bool = False,
) -> torch.Tensor:
    if dit.separated_timestep and fuse_vae_embedding_in_latents:
        spatial_tokens = latents.shape[3] * latents.shape[4] // 4
        timestep = torch.cat(
            [
                torch.zeros(
                    (1, spatial_tokens), dtype=latents.dtype, device=latents.device
                ),
                torch.ones(
                    (latents.shape[2] - 1, spatial_tokens),
                    dtype=latents.dtype,
                    device=latents.device,
                )
                * timestep,
            ]
        ).flatten()
        embedded_time = dit.time_embedding(
            sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0)
        )
    else:
        embedded_time = dit.time_embedding(
            sinusoidal_embedding_1d(dit.freq_dim, timestep)
        )

    if action_context is not None:
        if not hasattr(dit, "action_time_conditioner"):
            raise RuntimeError("world-model checkpoint is missing action conditioning")
        action = action_context.to(embedded_time)
        if action.shape[0] != embedded_time.shape[0]:
            if embedded_time.shape[0] % action.shape[0]:
                raise ValueError("action and timestep batches do not match")
            action = torch.cat(
                [action] * (embedded_time.shape[0] // action.shape[0]), dim=0
            )
        action_time = dit.action_time_conditioner(action)
        if embedded_time.ndim == 3:
            action_time = action_time.unsqueeze(1)
        embedded_time = embedded_time + action_time.to(embedded_time)

    if dit.separated_timestep and fuse_vae_embedding_in_latents:
        time_modulation = dit.time_projection(embedded_time).unflatten(
            2, (6, dit.dim)
        )
    else:
        time_modulation = dit.time_projection(embedded_time).unflatten(
            1, (6, dit.dim)
        )

    context = dit.text_embedding(context)
    if representation_context is not None:
        representation_context = representation_context.to(context)
        if representation_context.shape[0] != context.shape[0]:
            representation_context = torch.cat(
                [representation_context] * context.shape[0], dim=0
            )
        if representation_context.shape[-1] != context.shape[-1]:
            raise ValueError("DINOv3 and umT5 context widths do not match")
        context = torch.cat([representation_context, context], dim=1)

    if latents.shape[0] != context.shape[0]:
        latents = torch.cat([latents] * context.shape[0], dim=0)
    x = dit.patchify(latents)
    frames, height, width = x.shape[2:]
    x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
    frequencies = torch.cat(
        [
            dit.freqs[0][:frames]
            .view(frames, 1, 1, -1)
            .expand(frames, height, width, -1),
            dit.freqs[1][:height]
            .view(1, height, 1, -1)
            .expand(frames, height, width, -1),
            dit.freqs[2][:width]
            .view(1, 1, width, -1)
            .expand(frames, height, width, -1),
        ],
        dim=-1,
    ).reshape(frames * height * width, 1, -1).to(x.device)
    for block in dit.blocks:
        x = gradient_checkpoint_forward(
            block,
            use_gradient_checkpointing,
            x,
            context,
            time_modulation,
            frequencies,
        )
    x = dit.head(x, embedded_time)
    return dit.unpatchify(x, (frames, height, width))
