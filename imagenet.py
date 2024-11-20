import os
from torchvision import datasets, transforms
from MLclf import MLclf
# root = 'data/imagenet'
# def get_imagenet(root, train = True, transform = None, target_transform = None):
#     if train:
#         root = os.path.join(root, 'Caltech256')
#     else:
#         root = os.path.join(root, 'val')
#     return datasets.ImageFolder(root = root,
#                                transform = transform,
#                                target_transform = target_transform)


def get_dataset(dir, name):
    if name == 'mnist':
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.Grayscale(3), transforms.ToTensor()])
        train_dataset = datasets.MNIST(dir, train=True, download=True, transform=transform)
        eval_dataset = datasets.MNIST(dir, train=False, transform=transform)

    elif name == 'cifar':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = datasets.CIFAR10(dir, train=True, download=True,
                                         transform=transform_train)
        eval_dataset = datasets.CIFAR10(dir, train=False, transform=transform_test)
    elif name == 'svhn':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_dataset = datasets.SVHN(dir, split='extra', download=True, transform=transform_train)
        eval_dataset = datasets.SVHN(dir, split='test', download=True, transform=transform_test)
    elif name == "Tiny_imagenet":
        MLclf.tinyimagenet_download(Download=False)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        train_dataset, eval_dataset, test_dataset = MLclf.tinyimagenet_clf_dataset(ratio_train=0.7, ratio_val=0.2,
                                                                                   seed_value=777, shuffle=True,
                                                                                   transform=transform,
                                                                                   save_clf_data=True)
    elif name == "Mini_imagenet":
        MLclf.miniimagenet_download(Download=False)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        train_dataset, eval_dataset, test_dataset = MLclf.miniimagenet_clf_dataset(ratio_train=0.7, ratio_val=0.2,
                                                                                   seed_value=777, shuffle=True,
                                                                                   transform=transform,
                                                                                   save_clf_data=True)

    return train_dataset, eval_dataset
if __name__ == "__main__":

    get_dataset("\data", "Tiny_imagenet")