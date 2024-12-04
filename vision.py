import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms


feature = np.loadtxt("feature/resnet50_o16_feature.txt")
traindir = "data_tinyimagenet/tiny-imagenet-200/train"
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    ),
)

train_sampler = None

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=(train_sampler is None),
    num_workers=2,
    pin_memory=True,
    sampler=train_sampler,
)
gpu = 0
torch.cuda.set_device(gpu)
label = None
for i, (images, target) in enumerate(train_loader):
    if gpu is not None:
        target = target.cuda(gpu, non_blocking=True)
    if label is None:
        label = target
    else:
        label = torch.concatenate([label, target])
    # compute output
tsne = TSNE(n_components=2, perplexity = 29, random_state=777)
xtsne = tsne.fit_transform(feature)

















print("Have a nice day!")