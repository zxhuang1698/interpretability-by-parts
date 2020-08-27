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
from cub200 import *
from utils import *
from model import ResNet101, ResNet50

# number of attributes and landmark annotations
num_classes = 200

# arguments
parser = argparse.ArgumentParser(description='Evalution of Classification Accuracy')
parser.add_argument('--load', default='', type=str, help='name of model to evaluate')
parser.add_argument('--noaug', default='', action='store_true', help='disable test-time augmentation')
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
    acc: float
        Accuracy on the test set.

    """
    # set up the averagemeters
    batch_time = AverageMeter()
    acc = AverageMeter()

    # record the time
    end = time.time()

    # testing
    with torch.no_grad():
        for i, (input, target, _, _) in enumerate(test_loader):

            # data to gpu
            input = input.cuda()
            target = target.cuda()

            if args.noaug:
                # inference the model
                output, _, _ = model(input)
            
            else:
                # flip the data
                input_flip = torch.flip(input, dims=[3])

                # inference the model
                output, _, _ = model(input)
                output_flip, _, _ = model(input_flip)
                output = output + output_flip

            # calculate the accuracy
            acc.update(accuracy(output, target)[0].item(), input.size(0))

            # measure elapsed time
            torch.cuda.synchronize()
            batch_time.update(time.time() - end)
            end = time.time()

            # print the current testing status
            if i % 10 == 0:
                print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                    i, len(test_loader), batch_time=batch_time, acc=acc), flush=True)

    # return the accuracy
    return acc.avg

def main():

    # load the config file
    config_file = '../../log/'+ args.load +'/train_config.json'
    with open(config_file) as fi:
        config = json.load(fi)
        print(" ".join("\033[96m{}\033[0m: {},".format(k, v) for k, v in config.items()))

    # define data transformation
    test_transforms = transforms.Compose([
        transforms.Resize(size=448),
        transforms.CenterCrop(size=448),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
        ])

    # define test dataset and loader
    test_data = CUB200(root='../../data/cub200',
                       train=False, transform=test_transforms)

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
    acc = test(test_loader, model)

    # print the overall best acc
    print('Testing finished...')
    print('Best accuracy on test set is: %.4f.' % acc)

if __name__ == '__main__':
    main()
