# system lib
import argparse
import os
import shutil
import time
import random
import pdb
import numpy as np
import json

# pytorch lib
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.models as models

# import resnet101-based model
from model import ResNet101, ResNet50

# import utils and loss
import sys
sys.path.append(os.path.abspath('../common'))
from utils import *
from loss import *

# import tensorboardX for logging
from torch.utils.tensorboard.writer import SummaryWriter

# import dataset
from cub200 import *

# benchmark before running
cudnn.benchmark = True

# arguments for the script itself
parser = argparse.ArgumentParser(description='Interpretable network Training')
parser.add_argument('--config', default='../../cub_res101.json', type=str, help='path for the training config file')
parser.add_argument('--config-help', action='store_true', help='usage for each arguments in the config file')
args_main = parser.parse_args()

# print help information
if args_main.config_help is True:
    with open('../config_helper.json') as fh:
        helper = json.load(fh)
        print("\n".join("\033[96m{}\033[0m: {}".format(k, v) for k, v in helper.items()))
    exit()

# arguments for the training config
with open(args_main.config) as fi:
    args = json.load(fi)
    print(" ".join("\033[96m{}\033[0m: {},".format(k, v) for k, v in args.items()))


# fix all the randomness for reproducibility (for faster training and inference, please enable cudnn)
# torch.backends.cudnn.enabled = False
if args['seed'] != -1:
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed'])
    np.random.seed(args['seed'])
    random.seed(args['seed'])

# compose paths
data_dir = os.path.join(args['root'], "data", args['data'])
log_dir = os.path.join(args['root'], "log", args['save'])
check_dir = os.path.join(args['root'], "checkpoints")
tensorboard_dir = os.path.join(args['root'], "tensorboard_log", args['save'])

# check the existence of data
if os.path.exists(data_dir) is False:
    raise(RuntimeError("The data path does not exist!"))

# create the folder for log
if os.path.exists(log_dir) is False:
    os.makedirs(log_dir)
else:
    shutil.rmtree(log_dir)
    os.makedirs(log_dir)

# create the folder for checkpoint
if os.path.exists(check_dir) is False:
    os.makedirs(check_dir)

# create the folder for tensorboard writer
if os.path.exists(tensorboard_dir) is False:
    os.makedirs(tensorboard_dir)
else:
    shutil.rmtree(tensorboard_dir)
    os.makedirs(tensorboard_dir)

# init the tensorboard writer
writer = SummaryWriter(tensorboard_dir)

# copy config file to the log folder
shutil.copyfile(args_main.config, os.path.join(log_dir, "train_config.json"))

# global variable for accuracy
best_acc = 0

def main():

    global best_acc

    # create model by archetecture and load the pretrain weight
    print("=> creating model...")

    if args['arch'] == 'resnet101':
        model = ResNet101(args['num_classes'], args['nparts'])
        model.load_state_dict(models.resnet101(pretrained=True).state_dict(), strict=False)
    elif args['arch'] == 'resnet50':
        model = ResNet50(args['num_classes'], args['nparts'])
        model.load_state_dict(models.resnet50(pretrained=True).state_dict(), strict=False)
    else:
        raise(RuntimeError("Only support ResNet50 or ResNet101!"))

    model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    start_epoch = 0
    if args['resume'] != '':
        if os.path.isfile(args['resume']):
            print("=> loading checkpoint '{}'".format(args['resume']))
            checkpoint = torch.load(args['resume'])
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args['resume'], checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args['resume']))

    # data augmentation
    train_transforms = transforms.Compose([
        transforms.Resize(size=448),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1),
        transforms.RandomCrop(size=448),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    test_transforms = transforms.Compose([
        transforms.Resize(size=448),
        transforms.CenterCrop(size=448),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    # wrap to dataset
    train_data = CUB200(root=data_dir, train=True, transform=train_transforms)
    test_data = CUB200(root=data_dir, train=False, transform=test_transforms)

    # wrap to dataloader
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args['batch_size'], shuffle=True,
        num_workers=args['workers'], pin_memory=False, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args['batch_size'], shuffle=False,
        num_workers=args['workers'], pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # fix/finetune several layers
    fixed_layers = args['fixed']
    finetune_layers = args['finetune']
    finetune_parameters = []
    scratch_parameters = []
    for name, p in model.named_parameters():
        layer_name = name.split('.')[1]
        if layer_name not in fixed_layers:
            if layer_name in finetune_layers:
                finetune_parameters.append(p)
            else:
                scratch_parameters.append(p)
        else:
            p.requires_grad = False

    # define the optimizer according to different param groups
    optimizer = torch.optim.SGD([{'params': scratch_parameters,  'lr':20*args['lr']},
                                 {'params': finetune_parameters, 'lr':args['lr']},
        ], weight_decay=args['weight_decay'], momentum=0.9)

    # define the MultiStep learning rate scheduler
    num_iters = len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_iters * args['epochs'])

    # load the scheduler from the checkpoint if needed
    if args['resume'] != '':
        if os.path.isfile(args['resume']):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler'])

    # training part
    for epoch in range(start_epoch, args['epochs']):

        # training
        train(train_loader, model, criterion, optimizer, epoch, scheduler)

        # evaluate on test set
        acc = test(test_loader, model, criterion, epoch)

        # remember best acc and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'scheduler': scheduler.state_dict(),
        }, is_best, os.path.join(check_dir, args['save']))

        # print current best acc
        print('Current best average accuracy is: %.4f' % best_acc)

    # print the overall best acc and close the writer
    print('Training finished...')
    print('Best accuracy on test set is: %.4f.' % best_acc)
    writer.close()

def train(train_loader, model, criterion, optimizer, epoch, scheduler):

    # set up the averagemeters
    batch_time = AverageMeter()
    pred_losses = AverageMeter()
    shaping_losses = AverageMeter()
    acc = AverageMeter()
    num_iters = len(train_loader)

    # switch to train mode
    model.train()

    # record time
    end = time.time()

    # training step
    for i, (input, target, _, _) in enumerate(train_loader):

        # data to gpu
        input = input.cuda()
        target = target.cuda()

        # compute output
        output, _, assign = model(input)

        # compute the prediction loss and the shaping loss
        pred_loss = criterion(output, target)
        shaping_loss = ShapingLoss(assign, args['radius'], args['std'], args['nparts'], args['alpha'], args['beta'])

        # calculate the loss for bp
        loss = pred_loss + args['coeff'] * shaping_loss

        # record the losses and accuracy for tensorboard log
        acc1 = accuracy(output.data, target)[0]
        pred_losses.update(pred_loss.data.item(), input.size(0))
        shaping_losses.update(shaping_loss.item(), input.size(0))
        acc.update(acc1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print the current status
        if i % args['print_freq'] == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Cls_Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Shaping_Loss {shaping_loss.val:.4f} ({shaping_loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   loss=pred_losses, shaping_loss=shaping_losses, acc=acc), flush=True)
            # record the current loss status with tensorboard
            writer.add_scalar('data/pred_loss', pred_losses.val, epoch * num_iters + i)
            writer.add_scalar('data/shaping_loss', shaping_losses.val, epoch * num_iters + i)

    # print the learning rate
    lr = scheduler.get_lr()[0]
    print("Epoch {:d} finished with lr={:f}".format(epoch + 1, lr))
    # record the current accuracy status with tensorboard
    writer.add_scalars('data/accuracy', {"train" : acc.avg}, epoch + 1)

def test(test_loader, model, criterion, epoch):

    # set up the averagemeters
    batch_time = AverageMeter()
    pred_losses = AverageMeter()
    shaping_losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    # record the time
    end = time.time()

    # testing
    with torch.no_grad():
        for i, (input, target, _, _) in enumerate(test_loader):

            # data to gpu
            input = input.cuda()
            target = target.cuda()

            # inference the model
            output, _, assign = model(input)

            # compute the prediction loss and the shaping loss
            pred_loss = criterion(output, target)
            shaping_loss = ShapingLoss(assign, args['radius'], args['std'], args['nparts'], args['alpha'], args['beta'])

            # record the losses and accuracy for tensorboard log
            acc1 = accuracy(output.data, target)[0]
            pred_losses.update(pred_loss.data.item(), input.size(0))
            shaping_losses.update(shaping_loss.item(), input.size(0))
            acc.update(acc1.item(), input.size(0))

            # measure elapsed time
            torch.cuda.synchronize()
            batch_time.update(time.time() - end)
            end = time.time()

            # print the current testing status
            if i % args['print_freq'] == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Cls_Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Shaping_Loss {shaping_loss.val:.4f} ({shaping_loss.avg:.4f})\t'
                      'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                      i, len(test_loader), batch_time=batch_time,
                      loss=pred_losses, shaping_loss=shaping_losses, acc=acc), flush=True)

    # print the accuracy after the testing
    print(' \033[92m* Accuracy: {acc.avg:.3f}\033[0m'.format(acc=acc))

    # record the accuracy with tensorboard
    writer.add_scalars('data/accuracy', {"test" : acc.avg}, epoch + 1)

    # return the accuracy
    return acc.avg


if __name__ == '__main__':
    main()
