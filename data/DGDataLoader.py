import os
from torch.utils.data import DataLoader

from data.DGDataReader import *
from data.ConcatDataset import ConcatDataset
from utils.tools import *

default_input_dir = 'path/to/datalists/'

digits_datset = ["mnist", "mnist_m", "svhn", "syn"]
pacs_dataset = ["art_painting", "cartoon", "photo", "sketch"]
officehome_dataset = ['Art', 'Clipart', 'Product', 'Real_World']
available_datasets = pacs_dataset + officehome_dataset + digits_datset


def get_datalists_folder(args=None):
    datalists_folder = default_input_dir
    if args is not None:
        if args.input_dir is not None:
            datalists_folder = args.input_dir
    return datalists_folder


def get_train_dataloader(source_list=None, batch_size=64, image_size=224, crop=False, jitter=0, args=None, config=None):
    if args is not None:
        source_list = args.source
    if config is not None:
        batch_size = config["batch_size"]
        data_config = config["data_opt"]
    else:
        data_config = None
    assert isinstance(source_list, list)
    datasets = []
    for dname in source_list:
        datalists_folder = get_datalists_folder(args)
        path = os.path.join(datalists_folder, '%s_train.txt' % dname)
        train_dataset = get_dataset(path=path,
                                    train=True,
                                    image_size=image_size,
                                    crop=crop,
                                    jitter=jitter,
                                    config=data_config)
        datasets.append(train_dataset)
    dataset = ConcatDataset(datasets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False, drop_last=True)
    return loader


def get_fourier_train_dataloader(
        source_list=None,
        batch_size=64,
        image_size=224,
        crop=False,
        jitter=0,
        args=None,
        from_domain='all',
        alpha=1.0,
        config=None
):
    if args is not None:
        source_list = args.source
    if config is not None:
        batch_size = config["batch_size"]
        data_config = config["data_opt"]
    else:
        data_config = None
    assert isinstance(source_list, list)

    paths = []
    for dname in source_list:
        datalists_folder = get_datalists_folder(args)
        path = os.path.join(datalists_folder, '%s_train.txt' % dname)
        paths.append(path)
    dataset = get_fourier_dataset(path=paths,
                                  image_size=image_size,
                                  crop=crop,
                                  jitter=jitter,
                                  from_domain=from_domain,
                                  alpha=alpha,
                                  config=data_config)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False, drop_last=True)
    return loader


def get_val_dataloader(source_list=None, batch_size=64, image_size=224, args=None, config=None):
    if args is not None:
        source_list = args.source
    if config is not None:
        batch_size = config["batch_size"]
        data_config = config["data_opt"]
    else:
        data_config = None
    assert isinstance(source_list, list)
    datasets = []
    for dname in source_list:
        datalists_folder = get_datalists_folder(args)
        path = os.path.join(datalists_folder, '%s_val.txt' % dname)
        val_dataset = get_dataset(path=path, train=False, image_size=image_size, config=data_config)
        datasets.append(val_dataset)
    dataset = ConcatDataset(datasets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
    return loader


def get_test_loader(target=None, batch_size=64, image_size=224, args=None, config=None):
    if args is not None:
        target = args.target
    if config is not None:
        batch_size = config["batch_size"]
        data_config = config["data_opt"]
    else:
        data_config = None
    data_folder = get_datalists_folder(args)
    path = os.path.join(data_folder, '%s_test.txt' % target)
    test_dataset = get_dataset(path=path, train=False, image_size=image_size, config=data_config)
    dataset = ConcatDataset([test_dataset])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
    return loader

if __name__ == "__main__":
    batch_size=16
    source = ["art_painting", "cartoon", "photo"]
    loader = get_fourier_train_dataloader(source, batch_size, image_size=224, from_domain='all', alpha=1.0)

    it = iter(loader)
    batch = next(it)
    images = torch.cat(batch[0], dim=0)
    # images = batch[0][0]
    save_image_from_tensor_batch(images, batch_size, path='batch.jpg', device='cpu')