#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import random
import io
from torchvision import transforms


def diff_copyright_fix_subset(dataset, copyright_list):
    remain_list = list(set(range(len(dataset))) - set(copyright_list))
    remain_list = random.sample(remain_list, k=args.subset_size*2)  # ignore for POKEMON
    remain_data = dataset.select(remain_list)
    remain_shard0 = remain_data.shard(num_shards=2, index=0)
    remain_shard1 = remain_data.shard(num_shards=2, index=1)
    save_dir = os.path.join(args.save_dir, args.dataset_name)
    remain_shard0.save_to_disk(os.path.join(save_dir, f'q1_set'))
    remain_shard1.save_to_disk(os.path.join(save_dir, f'q2_set'))
 
    for img in copyright_list:
        image = Image.open(io.BytesIO(dataset[img]['image'])).convert("RGB")
        image.save(f'./target_imgs/{img}.jpg')
        image = Image.open(f'./target_imgs/{img}.jpg')
        image = train_transforms(image)
        image.save(f'./target_imgs/{img}.jpg')
    dup_num = int(len(remain_shard0) * 0.01)
    model1_set_list, model2_set_list = [], []
    for i in copyright_list:
        copyright_im = dataset.select([i])
        print('copyright image number = {}, caption: {}'.format(i, copyright_im[0]['caption']))  # 'caption' -> 'text' for POKEMON
        copyright_im = concatenate_datasets([copyright_im]*dup_num)
        model1_set = concatenate_datasets([copyright_im, remain_shard0])
        model2_set = concatenate_datasets([copyright_im, remain_shard1])
        print(model1_set)
        print(model2_set)
        model1_set_list.append(model1_set)
        model2_set_list.append(model2_set)
    return model1_set_list, model2_set_list



def download_image(url):
    try:
        response = requests.get(url, timeout=5) 
        response.raise_for_status()  
        return Image.open(BytesIO(response.content))
    except requests.exceptions.Timeout:
        print(f"Timeout occurred while downloading image from URL: {url}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error while downloading image from URL: {url}. Error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error occurred while downloading image from URL: {url}. Error: {e}")
        return None


def pil_image_to_bytes(pil_image):
    try:
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        img_byte_array = BytesIO()
        pil_image.save(img_byte_array, format='JPEG')
        return img_byte_array.getvalue()
    except Exception as e:
        print(f"Error converting image: {e}")
        return None


def get_laion_mi_set():
    laion_mi = pd.read_parquet('./laion_mi_non_members_metadata.parquet')

    failed_urls = pd.DataFrame(columns=laion_mi.columns)
    images = []
    print('Downloading images...')
    for index, row in tqdm(laion_mi.iterrows(), total=len(laion_mi)):
        url = row['url']
        image = download_image(url)
        if image is None:
            failed_urls = failed_urls.append(row, ignore_index=True)
        images.append(image)
        if index % args.slice_size == args.slice_size - 1:
            slice_id = index // args.slice_size
            slice_data = laion_mi[index + 1 - args.slice_size:index + 1].copy()
            slice_data['raw_image'] = images
            slice_data = slice_data[slice_data.raw_image.notnull()]
            slice_data['image'] = slice_data['raw_image'].apply(pil_image_to_bytes)
            slice_data = slice_data[slice_data.image.notnull()]
            slice_data.drop(columns=['raw_image'], inplace=True)
            print('Save slice_data {} with length {}...'.format(slice_id, len(slice_data)))
            slice_data.to_parquet(os.path.join(args.save_dir, args.dataset_name, 'sliced_data', f'laion_mi_{int(slice_id)}.parquet'), index=False)
            images = []

    failed_urls.to_csv('./failed_laion_mi_urls.csv', index=False, header=False)
    laion_mi['raw_image'] = images
    print('Original dataset length: {}'.format(len(laion_mi)))
    laion_mi = laion_mi[laion_mi.raw_image.notnull()]
    laion_mi['image'] = laion_mi['raw_image'].apply(pil_image_to_bytes)
    laion_mi = laion_mi[laion_mi.image.notnull()]
    laion_mi.drop(columns=['raw_image'], inplace=True)

    print('After dropping #{} of None images, dataset length: {}'.format(len(failed_urls), len(laion_mi)))
    dataset = Dataset.from_pandas(laion_mi)
    return dataset


def resume_laion_mi_dataset():
    silce_data_path = os.path.join(args.save_dir, args.dataset_name, 'sliced_data')
    laion_mi = pd.read_parquet(os.path.join(silce_data_path, 'laion_mi_0.parquet'))
    for i in range(1,13):
        slice_data = pd.read_parquet(os.path.join(silce_data_path, f'laion_mi_{int(i)}.parquet'))
        laion_mi = pd.concat([laion_mi, slice_data])
    dataset = Dataset.from_pandas(laion_mi)
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="evaluation script")
    parser.add_argument("--dataset_name", type=str, default='laion_mi',
                        help='select from pokemon, laion_mi')
    parser.add_argument("--subset_size", type=int, default=5000, help='size of subset')
    parser.add_argument("--slice_size", type=int, default=2000, help='number of each dataset slice size')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default='./datasets/')
    args = parser.parse_args()
    random.seed(args.seed)

    train_transforms = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            # transforms.ToTensor(),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    if 'pokemon' in args.dataset_name:
        dataset = load_dataset("lambdalabs/pokemon-blip-captions")['train']
    elif 'laion_mi' in args.dataset_name:
        data_path = os.path.join(args.save_dir, args.dataset_name, 'origin_laion_mi')
        if not os.path.exists(data_path):
            dataset = get_laion_mi_set()
            dataset = resume_laion_mi_dataset()
            dataset.save_to_disk(data_path)
        else:
            dataset = load_from_disk(data_path)
    else:
        dataset = None

    copyright_list = [5773]
   
    model1_set_list, model2_set_list = diff_copyright_fix_subset(dataset, copyright_list)
    save_dir = os.path.join(args.save_dir, args.dataset_name)
    for i in range(len(copyright_list)):
        img_no = copyright_list[i]
        print('Saving {}th image {} datasets'.format(i, img_no))
        model1_set_list[i].save_to_disk(os.path.join(save_dir, f'image_{img_no}_q1_set'))
        model2_set_list[i].save_to_disk(os.path.join(save_dir, f'image_{img_no}_q2_set'))
