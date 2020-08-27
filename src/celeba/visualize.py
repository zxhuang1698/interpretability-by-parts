# pytorch, vis and image libs
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg
from PIL import Image
import colorsys
import torch
import torch.nn as nn
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import cv2
import json
from collections import OrderedDict

# sys libs
import os
import argparse
import random

# dataset, utils and model
import sys
import os
sys.path.append(os.path.abspath('../common'))
from utils import *
from celeba import *
from model import ResNet101, ResNet50

# fix all the randomness for reproducibility
torch.backends.cudnn.enabled = False
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)

# number of attributes
num_classes = 40

# arguments
parser = argparse.ArgumentParser(description='Result Visualization')
parser.add_argument('--load', default='', type=str, help='name of model to visualize')
args = parser.parse_args()


def generate_colors(num_colors):
    """
    Generate distinct value by sampling on hls domain.

    Parameters
    ----------
    num_colors: int
        Number of colors to generate.

    Returns
    ----------
    colors_np: np.array, [num_colors, 3]
        Numpy array with rows representing the colors.

    """
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = 0.5
        saturation = 0.9
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    colors_np = np.array(colors)*255.

    return colors_np

def show_att_on_image(img, mask, output):
    """
    Convert the grayscale attention into heatmap on the image, and save the visualization.

    Parameters
    ----------
    img: np.array, [H, W, 3]
        Original colored image.
    mask: np.array, [H, W]
        Attention map normalized by subtracting min and dividing by max.
    output: str
        Destination image (path) to save.

    Returns
    ----------
    Save the result to output.

    """
    # generate heatmap and normalize into [0, 1]
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    # add heatmap onto the image
    merged = heatmap + np.float32(img)

    # re-scale the image
    merged = merged / np.max(merged)
    cv2.imwrite(output, np.uint8(255 * merged))

def plot_assignment(root, assign_hard, num_parts):
    """
    Blend the original image and the colored assignment maps.

    Parameters
    ----------
    root: str
        Root path for saving visualization results.
    assign_hard: np.array, [H, W]
        Hard assignment map (int) denoting the deterministic assignment of each pixel. Generated via argmax.
    num_parts: int, number of object parts.

    Returns
    ----------
    Save the result to root/assignment.png.

    """
    # generate the numpy array for colors
    colors = generate_colors(num_parts)

    # coefficient for blending
    coeff = 0.4

    # load the input as RGB image, convert into numpy array
    input = Image.open(os.path.join(root, 'input.png')).convert('RGB')
    input_np = np.array(input).astype(float)

    # blending by each pixel
    for i in range(assign_hard.shape[0]):
        for j in range(assign_hard.shape[1]):
            assign_ij = assign_hard[i][j]
            input_np[i, j] = (1-coeff) * input_np[i, j] + coeff * colors[assign_ij]

    # save the resulting image
    im = Image.fromarray(np.uint8(input_np))
    im.save(os.path.join(root, 'assignment.png'))

def main():

    # load the config file
    config_file = '../../log/'+ args.load +'/train_config.json'
    with open(config_file) as fi:
        config = json.load(fi)
        print(" ".join("\033[96m{}\033[0m: {},".format(k, v) for k, v in config.items()))

    # define data transformation (no crop)
    test_transforms = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                std=(0.229, 0.224, 0.225))
        ])

    # define test dataset and loader
    if config['split'] == 'accuracy':
        dataset = CelebA('../../data/celeba', split='test', align=True,
            percentage=None, transform=test_transforms, resize=(256, 256))
    elif config['split'] == 'interpretability':
        dataset = CelebA('../../data/celeba', split='test', align=False,
            percentage=0.3, transform=test_transforms, resize=(256, 256))
    else:
        raise(RuntimeError("Please choose either \'accuracy\' or \'interpretability\' for data split."))
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1, shuffle=True,
        num_workers=1, pin_memory=False)

    # create a dataloader iter instance
    test_loader_iter = iter(test_loader)

    # define the figure layout
    fig_rows = 5
    fig_cols = 5
    f_assign, axarr_assign = plt.subplots(fig_rows, fig_cols, figsize=(fig_cols*2,fig_rows*2))
    f_assign.subplots_adjust(wspace=0, hspace=0)

    # load the model in eval mode
    # with batch size = 1, we only support single GPU visaulization
    if config['arch'] == 'resnet101':
        model = ResNet101(num_classes, num_parts=config['nparts']).cuda()
    elif config['arch'] == 'resnet50':
        model = ResNet50(num_classes, num_parts=config['nparts']).cuda()
    else:
        raise(RuntimeError("Only support resnet50 or resnet101 for architecture!"))

    # load model
    resume = '../../checkpoints/'+args.load+'_best.pth.tar'
    print("=> loading checkpoint '{}'".format(resume))
    checkpoint = torch.load(resume)
    # remove the module prefix
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    with torch.no_grad():
        # the visualization code
        current_id = 0
        for col_id in range(fig_cols):
            for j in range(fig_rows):

                # inference the model
                input, target, _ = next(test_loader_iter)
                input = input.cuda()
                target = target.cuda()
                current_id += 1
                with torch.no_grad():
                    print("Visualizing %dth image..." % current_id)
                    output_list, att_list, assign = model(input)

                # define root for saving results and make directories correspondingly
                root = os.path.join('../../visualization', args.load, str(current_id))
                os.makedirs(root, exist_ok=True)
                os.makedirs(os.path.join(root, 'attentions'), exist_ok=True)
                os.makedirs(os.path.join(root, 'assignments'), exist_ok=True)

                # denormalize the image and save the input
                save_input = transforms.Normalize(mean=(0, 0, 0),std=(1/0.229, 1/0.224, 1/0.225))(input.data[0].cpu())
                save_input = transforms.Normalize(mean=(-0.485, -0.456, -0.406),std=(1, 1, 1))(save_input)
                save_input = torch.nn.functional.interpolate(save_input.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
                img = torchvision.transforms.ToPILImage()(save_input)
                img.save(os.path.join(root, 'input.png'))

                # save the labels and pred as list
                label = list(target.data[0].cpu().numpy())
                prediction = []
                assert (len(label) == num_classes)
                for k in range(num_classes):
                    current_score = torch.sigmoid(output_list[k]).squeeze().data.item()
                    current_pred = int(current_score > 0.5)
                    prediction.append(current_pred)

                # write the labels and pred
                with open(os.path.join(root, 'prediction.txt'), 'w') as pred_log:
                    for k in range(num_classes):
                        pred_log.write('%s pred: %d, label: %d\n' % (celeba_attr[k], prediction[k], label[k]))

                # upsample the assignment and transform the attention correspondingly
                assign_reshaped = torch.nn.functional.interpolate(assign.data.cpu(), size=(256, 256), mode='bilinear', align_corners=False)

                # visualize the attention
                for k in range(num_classes):

                    # attention vector for kth attribute
                    att = att_list[k].view(
                        1, config['nparts'], 1, 1).data.cpu()

                    # multiply the assignment with the attention vector
                    assign_att = assign_reshaped * att

                    # sum along the part dimension to calculate the spatial attention map
                    attmap_hw = torch.sum(assign_att, dim=1).squeeze(0).numpy()

                    # normalize the attention map and merge it onto the input
                    img = cv2.imread(os.path.join(root, 'input.png'))
                    mask = attmap_hw / attmap_hw.max()
                    img_float = img.astype(float) / 255.
                    show_att_on_image(img_float, mask, os.path.join(root, 'attentions', celeba_attr[k]+'.png'))

                # generate the one-channel hard assignment via argmax
                _, assign = torch.max(assign_reshaped, 1)

                # colorize and save the assignment
                plot_assignment(root, assign.squeeze(0).numpy(), config['nparts'])

                # collect the assignment for the final image array
                color_assignment_name = os.path.join(root, 'assignment.png')
                color_assignment = mpimg.imread(color_assignment_name)
                axarr_assign[j, col_id].imshow(color_assignment)
                axarr_assign[j, col_id].axis('off')

                # plot the assignment for each dictionary vector
                for i in range(config['nparts']):
                    img = torch.nn.functional.interpolate(assign_reshaped.data[:, i].cpu().unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False)
                    img = torchvision.transforms.ToPILImage()(img.squeeze(0))
                    img.save(os.path.join(root, 'assignments', 'part_'+str(i)+'.png'))

        # save the array version
        os.makedirs('../../visualization/collected', exist_ok=True)
        f_assign.savefig(os.path.join('../../visualization/collected', args.load+'.png'))

        print('Visualization finished!')

# main method
if __name__ == '__main__':
    main()
