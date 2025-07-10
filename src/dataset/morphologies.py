import numpy as np
import PIL.Image as Image
import json
from pathlib import Path


class CompositionalMorpologies:
    def __init__(
        self,
        dataset_folder: str | Path,
        image_size: tuple[int, int] = (64, 64)
    ) -> None:
        super().__init__()

        dataset_folder = Path(dataset_folder)
        with open(dataset_folder / 'metadata.json') as f:
            metadata_info = json.load(f)
            locals().update(metadata_info)

        self.image_files = sorted(dataset_folder.glob("*.png"))
        self.image_size = image_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = load_image(image_file, self.image_size)
        return int(image_file.stem), image.transpose((2, 0, 1))


class SingleCompositeMorphology:
    def __init__(self,
        dataset_folder: str | Path,
        image_size=(64, 64),
        shape_idx=0,
        batch_size=8
    ) -> None:
        super().__init__()
        self.batch_size = batch_size

        dataset_folder = Path(dataset_folder)
        with open(dataset_folder / 'metadata.json') as f:
            metadata_info = json.load(f)
            locals().update(metadata_info)

        image_file = sorted(dataset_folder.glob("*.png"))[shape_idx]

        # image = np.asarray(Image.open(image_file.resolve()).convert('RGBA')).copy()
        # image = resize(image, image_size, anti_aliasing=True).astype(np.float32)
        # # image[..., :3] *= image[..., 3:]
        # self.image = image

        self.image = load_image(image_file, image_size)
        self.image_name = image_file.stem

    def __getitem__(self, index):
        image = self.image.transpose((2, 0, 1))
        return int(self.image_name), np.repeat(image[None], self.batch_size, axis=0)


def load_image(image_file, image_size):
    image = Image.open(image_file.resolve()).convert('RGBA')
    image.thumbnail(image_size, Image.LANCZOS, reducing_gap=None)
    image = np.asarray(image, dtype=np.float32) / 255.0
    image[..., 3:] = image[..., 3:] * (image[..., 3:] > 0.1)  # remove some artifacts of the downscaling
    image[..., :3] *= image[..., 3:]
    return image
