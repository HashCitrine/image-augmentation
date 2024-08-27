import argparse
import os
from datetime import datetime
from pathlib import Path

import cv2
import imgaug.augmenters as iaa
import numpy as np
from deprecated import deprecated


def save_images(samples, type, save_path_str):
    save_path = Path(save_path_str)

    for i, sample in enumerate(samples):
        result_dir = save_path / f'{date_str}({file_origin_name})'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        # Not Use Augmentation Type
        # result_dir = result_dir + f"/{type}"
        # if not os.path.exists(result_dir):
        #     os.mkdir(result_dir)

        result_path = f"{result_dir}/{file_name}({i + 1}).{file_ext}"
        cv2.imwrite(result_path, sample)


@deprecated
def soft_augmentation(images):
    seq = iaa.Sequential([
        iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
        iaa.Fliplr(0.5),  # horizontally flip 50% of the images
        iaa.GaussianBlur(sigma=(0, 3.0))  # blur images with a sigma of 0 to 3.0
    ])

    images_aug = seq(images=images)
    save_images(images_aug, 'soft')


def standard_augmentation(images, save_path):
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Crop(percent=(0, 0.1)),  # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(
            0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.LinearContrast((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        )
    ], random_order=True)  # apply augmenters in random order

    images_aug = seq(images=images)
    save_images(images_aug, 'standard', save_path)


def init_arg():
    # ArgumentParser 객체 생성
    arg_parser = argparse.ArgumentParser(description="Process some integers.")

    # 이름이 붙은 인자 추가
    arg_parser.add_argument("-c", "--count", default=10, type=int, help="Number of times to augment")
    arg_parser.add_argument("-f", "--files", nargs='+', type=str, help="File path of the image you want to augment")
    arg_parser.add_argument("-s", "--save", default='./', type=str, help="Save path for augmented images")
    return arg_parser


if __name__ == "__main__":
    parser = init_arg()
    args = parser.parse_args()

    file_path_arr = args.files
    augmentation_count = args.count
    save_augment_path = args.save

    for file_path in file_path_arr:
        image = cv2.imread(file_path)

        file_origin_name = os.path.basename(file_path)
        file_name_arr = file_origin_name.split('.')
        file_name = file_name_arr[0]
        file_ext = file_name_arr[1]
        date_str = datetime.now().strftime('%Y%m%d_%H%M%S')

        images = np.array(
            [image for _ in range(augmentation_count)],
            dtype=np.uint8
        )
        print(f"[{file_origin_name}] Working...")

        standard_augmentation(images, save_augment_path)

        print(f"[{file_origin_name}] Done")
