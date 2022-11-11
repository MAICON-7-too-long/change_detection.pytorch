import os.path as osp

import albumentations as A
from albumentations.pytorch import ToTensorV2

from .custom import CustomDataset
from .transforms.albu import ChunkImage, ToTensorTest


class MAICON_Dataset(CustomDataset):
    """MAICON_Dataset"""

    def __init__(self, img_dir, sub_dir_1='input1', sub_dir_2='input2', ann_dir=None, img_suffix='.png', seg_map_suffix='.png', transform=None, split=None, data_root=None, test_mode=False, size=256, debug=False):
        super().__init__(img_dir, sub_dir_1, sub_dir_2, ann_dir, img_suffix, seg_map_suffix, transform, split, data_root, test_mode, size, debug)

    def get_default_transform(self):
        """Set the default transformation."""

        default_transform = A.Compose([
            A.RandomCrop(height=512, width=512, p=0.3),
            A.PixelDropout(p=0.1),
            A.RandomGamma(p=0.2),
            A.OneOf([
                A.Blur(blur_limit=3, p=1),
                A.Defocus(p=0.5),
                A.Spatter(p=1),
                A.Emboss(p=0.7),
            ], p=0.1),
            A.OneOf([
                A.CLAHE(p=0.1),
                A.RandomBrightnessContrast(p=0.3),
                A.HueSaturationValue(p=0.1),
                A.RandomToneCurve(p=0.2),
            ], p=0.2),
            A.Downscale(scale_min=0.5, scale_max=0.8, p=0.2),
            A.OneOf([
                A.Flip(p=0.1),
                A.ShiftScaleRotate(p=0.05),
                A.Perspective(p=0.1),
                A.GridDistortion(p=0.3),
            ], p=0.1),
            A.OneOf([
                A.GaussNoise(p=0.5),
                A.ISONoise(p=0.5),
                A.MultiplicativeNoise(p=0.5),
            ], p=0.3),
            A.Resize(self.size, self.size),
            A.Normalize(),
            ToTensorV2()
        ], additional_targets={'image_2': 'image'})
        return default_transform

    def get_test_transform(self):
        """Set the test transformation."""

        test_transform = A.Compose([
            A.Resize(1024, 1024),
            A.Normalize(),
            ToTensorV2()
        ], additional_targets={'image_2': 'image'})
        return test_transform

    def __getitem__(self, idx):
        """Get training/test data after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if not self.ann_dir:
            ann = None
            img1, img2, filename = self.prepare_img(idx)
            size = img1.size
            transformed_data = self.transform(image=img1, image_2=img2)
            img1, img2 = transformed_data['image'], transformed_data['image_2']
            return img1, img2, filename
        else:
            img1, img2, ann, filename = self.prepare_img_ann(idx)
            transformed_data = self.transform(image=img1, image_2=img2, mask=ann)
            img1, img2, ann = transformed_data['image'], transformed_data['image_2'], transformed_data['mask']
            return img1, img2, ann, filename


if __name__ == "__main__":
    MAICON_Dataset('dir')
