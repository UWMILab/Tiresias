from PIL import Image

from lcwm.datasets.video import contain_and_pad


class LetterboxResize:
    def __init__(
        self,
        height: int = 256,
        width: int = 256,
        height_division_factor: int = 1,
        width_division_factor: int = 1,
    ) -> None:
        if height % height_division_factor or width % width_division_factor:
            raise ValueError("image size must satisfy the model division factors")
        self.height = int(height)
        self.width = int(width)

    def __call__(self, image: Image.Image) -> Image.Image:
        return contain_and_pad(image, (self.width, self.height))
