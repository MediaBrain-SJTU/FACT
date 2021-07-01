## FACT

This repo provides a demo for the CVPR 2021 paper "A Fourier-based Framework for Domain Generalization" on the PACS dataset.

To cite, please use:

```
@InProceedings{Xu_2021_CVPR,
    author    = {Xu, Qinwei and Zhang, Ruipeng and Zhang, Ya and Wang, Yanfeng and Tian, Qi},
    title     = {A Fourier-Based Framework for Domain Generalization},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {14383-14392}
}
```

#### Requirements

* `Python 3.6`
* `Pytorch 1.1.0`

#### Evaluation

Firstly create directory `ckpt/` and drag your model under it.  For running the evaluation code, please download the PACS dataset from http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017. Then update the files with suffix `_test.txt` in `data/datalists` for each domain, following styles below:

```
/home/user/data/images/PACS/kfold/art_painting/dog/pic_001.jpg 0
/home/user/data/images/PACS/kfold/art_painting/dog/pic_002.jpg 0
/home/user/data/images/PACS/kfold/art_painting/dog/pic_003.jpg 0
...
```

Once the data is prepared, remember to update the path of test files and output logs in `shell_test.py`:

``` 
input_dir = 'path/to/test/files'
output_dir = 'path/to/output/logs'
```

then simply run:

```
 python shell_test.py -d=art_painting
```

You can use the argument `-d` to specify the held-out target domain.

#### Training from scratch 

After downloading the dataset, update the files with suffix `_train.txt` and `_val.txt` in `data/datalists` for each domain, following the same styles as the test files above. Please make sure you are using the official train-val-split.  Then update the  the path of train&val files and output logs in `shell_train.py`:

```
input_dir = 'path/to/train/files'
output_dir = 'path/to/output/logs'
```

Then running the code:

```
python shell_train.py -d=art_painting
```

Use the argument `-d` to specify the held-out target domain.

#### Acknowledgement

Part of our code is borrowed from the following repositories.

* [JigenDG](https://github.com/fmcarlucci/JigenDG): "Domain Generalization by Solving Jigsaw Puzzles", CVPR 2019
* [DDAIG](https://github.com/KaiyangZhou/DG-research-pytorch): "Deep Domain-Adversarial Image Generation for Domain Generalisation", AAAI 2020

We thank to the authors for releasing their codes. Please also consider citing their works.