# pylint: disable=invalid-name, line-too-long, missing-function-docstring, missing-module-docstring

import os
import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from evaluate import evaluate
from utils.data_loading import BasicDataset, BirdseyeDataset
from unet import UNet

dir_img = Path('./data/test/imgs/')
dir_mask = Path('./data/test/masks/')

def test_model(model, device, batch_size, img_scale, amp):
    # Create test dataset
    test_set = BirdseyeDataset(dir_img, dir_mask, img_scale)

    # Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=True, **loader_args)

    # Iterate over the dataset and get the score
    test_score = evaluate(model, test_loader, device, amp)
    print(f"Test dice score: {test_score}")

def get_args():
    parser = argparse.ArgumentParser(description="Test UNet on unseen data")
    parser.add_argument("--model", "-m", metavar="M", type=str, default=False, help="Model file for testing")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(n_channels=3, n_classes=5, bilinear=False)
    model = model.to(memory_format=torch.channels_last)

    state_dict = torch.load(args.model, map_location=device)
    del state_dict['mask_values']
    model.load_state_dict(state_dict)

    model.to(device=device)

    try:
        test_model(model=model, device=device, batch_size=1, img_scale=1.0, amp=False)
    except Exception as e:
        print(e)
