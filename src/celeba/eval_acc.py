# pytorch & misc
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse
import time
import json

# dataset and model
import sys
import os
sys.path.append(os.path.abspath('../common'))
from celeba import *
from utils import *
from model import ResNet101, ResNet50

# number of attributes and landmark annotations
num_classes = 40

# arguments
parser = argparse.ArgumentParser(description='Evalution of Classification Accuracy')
parser.add_argument('--load', default='', type=str, help='name of model to evaluate')
args = parser.parse_args()

def test(test_loader, model):
    """
    Evaluate the accuracy on test set.

    Parameters
    ----------
    test_loader: torch.utils.data.DataLoader
        Data loader for the testing set.
    model: pytorch model object
        Model that generates attribute prediction.

    Returns
    ----------
    accs: list of 40 AverageMeter objects
        Per-attribute accuracy.
    avgacc_mean: float
        Average accuracy over all attributes

    """
    # set up the averagemeters
    batch_time = AverageMeter()
    accs = [AverageMeter() for _ in range(num_classes)]

    # record the time
    end = time.time()

    # testing
    with torch.no_grad():
        for i, (input, target, _) in enumerate(test_loader):

            # data to gpu
            input = input.cuda()
            target = target.cuda()

            # inference the model
            output, _, _ = model(input)

            # list to store prediction loss & accuracy by attribute
            acc = []
            for j in range(len(output)):

                # prediction accuracy for jth attribute
                acc.append(accuracy_binary(output[j], target[:, j]))

                # keep track of per category accuracy
                accs[j].update(acc[j].item(), input.size(0))

            # calculate mean (categories) acc for the current batch
            acc_mean = sum([acc[k].data.item() for k in range(len(output))]) / len(output)

            # calculate the mean (categories) value of averaged (batches so far) acc
            avgacc_mean = sum([accs[k].avg for k in range(len(output))]) / len(output)

            # measure elapsed time
            torch.cuda.synchronize()
            batch_time.update(time.time() - end)
            end = time.time()

            # print the current testing status
            if i % 50 == 0:
                print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Avg_acc {acc_val:.3f} ({acc_avg:.3f})'.format(
                    i, len(test_loader), batch_time=batch_time, acc_val=acc_mean, acc_avg=avgacc_mean), flush=True)

    # return the accuracy
    return accs, avgacc_mean

def main():

    # load the config file
    config_file = '../../log/'+ args.load +'/train_config.json'
    with open(config_file) as fi:
        config = json.load(fi)
        print(" ".join("\033[96m{}\033[0m: {},".format(k, v) for k, v in config.items()))

    # test transform
    test_transforms = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    # define test dataset and loader
    if config['split'] == 'accuracy':
        test_data = CelebA('../../data/celeba', split='test', align=True,
            percentage=None, transform=test_transforms, resize=(256, 256))
    elif config['split'] == 'interpretability':
        test_data = CelebA('../../data/celeba', split='test', align=False,
            percentage=0.3, transform=test_transforms, resize=(256, 256))
    else:
        raise(RuntimeError("Please choose either \'accuracy\' or \'interpretability\' for data split."))
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=config['batch_size'], shuffle=False,
        num_workers=config['workers'], pin_memory=False, drop_last=False)

    # load the model in eval mode
    if config['arch'] == 'resnet101':
        model = nn.DataParallel(ResNet101(num_classes, num_parts=config['nparts'])).cuda()
    elif config['arch'] == 'resnet50':
        model = nn.DataParallel(ResNet50(num_classes, num_parts=config['nparts'])).cuda()
    else:
        raise(RuntimeError("Only support resnet50 or resnet101 for architecture!"))

    resume = '../../checkpoints/'+args.load+'_best.pth.tar'
    print("=> loading checkpoint '{}'".format(resume))
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()

    # test the model
    acc_per_attr, acc = test(test_loader, model)

    # print the overall best acc
    print('Testing finished...')
    print('Per-attribute accuracy:')
    print('===========================================================================')
    for k in range(num_classes):
        print('\033[96m%s\033[0m: %.4f' % (celeba_attr[k], acc_per_attr[k].avg))
    print('===========================================================================')
    print('Best average accuracy on test set is: %.4f.' % acc)

if __name__ == '__main__':
    main()
