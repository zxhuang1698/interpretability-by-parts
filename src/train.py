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
from utils import *
from loss import *

# import tensorboardX for logging
from torch.utils.tensorboard.writer import SummaryWriter

# import dataset
from datasets.celeba import *

# benchmark before running
cudnn.benchmark = True

# arguments for the script itself
parser = argparse.ArgumentParser(description='Interpretable network Training')
parser.add_argument('--config', default='../cub_config.json', type=str, help='path for the training config file')
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

# create the folder for dataset
if os.path.exists(data_dir) is False:
    os.makedirs(data_dir)

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
        transforms.Resize(size=(256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    val_transforms = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    # wrap to dataset
    if args['split'] == 'accuracy':
        train_data = CelebA(data_dir, split='train_full', align=True,
        percentage=None, transform=train_transforms, resize=(256, 256))
        val_data = CelebA(data_dir, split='val', align=True,
        percentage=None, transform=val_transforms, resize=(256, 256))
    elif args['split'] == 'interpretability':
        train_data = CelebA(data_dir, split='train', align=False,
        percentage=0.3, transform=train_transforms, resize=(256, 256))
        val_data = CelebA(data_dir, split='val', align=False,
        percentage=0.3, transform=val_transforms, resize=(256, 256))
    else:
        raise(RuntimeError("Please choose either \'accuracy\' or \'interpretability\' for data split."))

    # wrap to dataloader
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args['batch_size'], shuffle=True,
        num_workers=args['workers'], pin_memory=False, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args['batch_size'], shuffle=False,
        num_workers=args['workers'], pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.BCEWithLogitsLoss().cuda()

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
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5], gamma=0.1)

    # load the scheduler from the checkpoint if needed
    if args['resume'] != '':
        if os.path.isfile(args['resume']):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler'])

    # training part
    for epoch in range(start_epoch, args['epochs']):

        # training
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on val set
        acc_per_attr, acc = validate(val_loader, model, criterion, epoch)

        # LR scheduler
        scheduler.step()

        # remember best acc and save checkpoint
        is_best = acc > best_acc
        if is_best:
            best_acc = acc
            best_per_attr = acc_per_attr
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
    with open(os.path.join(log_dir, "acc_per_attr.txt"), 'w') as logfile:
        for k in range(args['num_classes']):
            logfile.write('%s: %.4f\n' % (celeba_attr[k], best_per_attr[k].avg))
    print('Per-attribute accuracy on val set has been written to acc_per_attr.txt under the log folder')
    print('Best average accuracy on val set is: %.4f.' % best_acc)

    writer.close()

def train(train_loader, model, criterion, optimizer, epoch):

    # set up the averagemeters
    batch_time = AverageMeter()
    pred_losses = [AverageMeter() for _ in range(args['num_classes'])]
    shaping_losses = AverageMeter()
    accs = [AverageMeter() for _ in range(args['num_classes'])]
    num_iters = len(train_loader)

    # switch to train mode
    model.train()

    # record time
    end = time.time()

    # training step
    for i, (input, target, _) in enumerate(train_loader):

        # data to gpu
        input = input.cuda()
        target = target.cuda()

        # compute output
        output, _, assign = model(input)

        # compute the prediction loss accuracy by attribute
        pred_loss = []
        acc = []
        for j in range(len(output)):

            # prediction loss and accuracy for jth attribute
            pred_loss.append(criterion(output[j].squeeze(1), target[:, j].float()))
            acc.append(accuracy(output[j], target[:, j]))

            # keep track of the prediction loss and accuracy by categories
            pred_losses[j].update(pred_loss[j].data.item(), input.size(0))
            accs[j].update(acc[j].item(), input.size(0))

        # for tensorboard log
        # calculate mean (categories) value of loss and acc for the current batch
        losses_mean = sum([pred_loss[k].data.item() for k in range(len(output))]) / len(output)
        acc_mean = sum([acc[k].data.item() for k in range(len(output))]) / len(output)

        # calculate the mean (categories) value of averaged (batches so far) loss and acc
        avglosses_mean = sum([pred_losses[k].avg for k in range(len(output))]) / len(output)
        avgacc_mean = sum([accs[k].avg for k in range(len(output))]) / len(output)

        # calculate the loss for bp
        pred_loss_mean = sum(pred_loss) / len(pred_loss)
        shaping_loss = ShapingLoss(assign, args['radius'], args['std'], args['nparts'], args['alpha'], args['beta'])
        loss = pred_loss_mean + args['coeff'] * shaping_loss

        # record the shaping loss for tensorboard log
        shaping_losses.update(shaping_loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print the current status
        if i % args['print_freq'] == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Cls_Loss {loss_val:.4f} ({loss_avg:.4f})\t'
                  'Shaping_Loss {shaping_loss.val:.4f} ({shaping_loss.avg:.4f})\t'
                  'Avg_acc {acc_val:.3f} ({acc_avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   loss_val=losses_mean, loss_avg=avglosses_mean, shaping_loss=shaping_losses, acc_val=acc_mean, acc_avg=avgacc_mean), flush=True)
            # record the current loss status with tensorboard
            writer.add_scalar('data/pred_loss', losses_mean, epoch * num_iters + i)
            writer.add_scalar('data/shaping_loss', shaping_losses.val, epoch * num_iters + i)

    # record the current accuracy status with tensorboard
    writer.add_scalars('data/accuracy', {"train" : avgacc_mean}, epoch + 1)

def validate(val_loader, model, criterion, epoch):

    # set up the averagemeters
    batch_time = AverageMeter()
    pred_losses = [AverageMeter() for _ in range(args['num_classes'])]
    shaping_losses = AverageMeter()
    accs = [AverageMeter() for _ in range(args['num_classes'])]

    # switch to evaluate mode
    model.eval()

    # record the time
    end = time.time()

    # validating
    with torch.no_grad():
        for i, (input, target, _) in enumerate(val_loader):

            # data to gpu
            input = input.cuda()
            target = target.cuda()

            # inference the model
            output, _, assign = model(input)

            # list to store prediction loss & accuracy by attribute
            pred_loss = []
            acc = []
            for j in range(len(output)):

                # prediction loss and accuracy for jth attribute
                pred_loss.append(criterion(output[j].squeeze(1), target[:, j].float()))
                acc.append(accuracy(output[j], target[:, j]))

                # keep track of the prediction loss and accuracy by categories
                pred_losses[j].update(pred_loss[j].data.item(), input.size(0))
                accs[j].update(acc[j].item(), input.size(0))

            # for tensorboard log
            # calculate mean (categories) value of loss and acc for the current batch
            losses_mean = sum([pred_loss[k].data.item() for k in range(len(output))]) / len(output)
            acc_mean = sum([acc[k].data.item() for k in range(len(output))]) / len(output)

            # calculate the mean (categories) value of averaged (batches so far) loss and acc
            avglosses_mean = sum([pred_losses[k].avg for k in range(len(output))]) / len(output)
            avgacc_mean = sum([accs[k].avg for k in range(len(output))]) / len(output)

            # calculate the losses
            pred_loss_mean = sum(pred_loss) / len(pred_loss)
            shaping_loss = ShapingLoss(assign, args['radius'], args['std'], args['nparts'], args['alpha'], args['beta'])

            # record the shaping loss for tensorboard log
            shaping_losses.update(shaping_loss.item(), input.size(0))

            # measure elapsed time
            torch.cuda.synchronize()
            batch_time.update(time.time() - end)
            end = time.time()

            # print the current validation status
            if i % args['print_freq'] == 0:
                print('Val: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Cls_Loss {loss_val:.4f} ({loss_avg:.4f})\t'
                    'Shaping_Loss {shaping_loss.val:.4f} ({shaping_loss.avg:.4f})\t'
                    'Avg_acc {acc_val:.3f} ({acc_avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss_val=losses_mean,
                    loss_avg=avglosses_mean, shaping_loss=shaping_losses,
                    acc_val=acc_mean, acc_avg=avgacc_mean), flush=True)

    # print the accuracy after the validating
    print(' \033[92m* Accuracy: {acc_avg:.3f}\033[0m'.format(acc_avg=avgacc_mean))

    # record the accuracy with tensorboard
    writer.add_scalars('data/accuracy', {"val" : avgacc_mean}, epoch + 1)

    # return the accuracy
    return accs, avgacc_mean


if __name__ == '__main__':
    main()
