"""Generate benchmarking vnnlib files for trained models."""

import os
import shutil
import argparse
import csv
import tqdm
import random

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from vnnlib_utils import create_input_bounds, save_vnnlib


models = [
    {
        'name': 'vit_2_6',
        'num_instances': 5,
        'timeout': 300,
    }
]


def create_vnnlib(X, y, mean, std, index, path, eps=1./255):
    input_bounds = create_input_bounds(X[index], eps, mean, std)
    save_vnnlib(input_bounds, y[index], path, total_output_class=10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('seed', type=int)
    args = parser.parse_args()

    random.seed(args.seed)

    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2023, 0.1994, 0.2010])
    transform = transforms.Compose([transforms.ToTensor()])
    test_data = datasets.CIFAR10('./data', train=False, download=True,
                                 transform=transform)
    test_data = torch.utils.data.DataLoader(
        test_data, batch_size=10000, pin_memory=True, num_workers=4)
    X, y = next(iter(test_data))

    for file in ['vnnlib', 'onnx']:
        if os.path.exists(file):
            shutil.rmtree(file)
            os.makedirs(file)

    instances = []
    for model in models:
        name = model['name']
        path = f'models/{name}'
        with open(f'{path}/index.txt') as f:
            indexes = list(map(int, f.readlines()))
        random.shuffle(indexes)
        onnx_path = f'onnx/{name}.onnx'
        shutil.copy(f'{path}/model.onnx', onnx_path)
        for i in range(model['num_instances']):
            vnnlib_path = f'vnnlib/{name}_{indexes[i]}.vnnlib'
            create_vnnlib(X, y, mean, std, indexes[i], vnnlib_path)
            instances.append((onnx_path, vnnlib_path, model['timeout']))

    with open('instances.csv', 'w') as f:
        csv.writer(f).writerows(instances)
