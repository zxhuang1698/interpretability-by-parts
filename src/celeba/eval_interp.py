# pytorch & misc
import numpy as np
from PIL import Image, ImageDraw
from numpy.linalg import norm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import cv2
import argparse
import json

# dataset and model
from celeba import *
from model import ResNet101, ResNet50

# number of attributes and landmark annotations
num_classes = 40
num_landmarks = 5

# arguments
parser = argparse.ArgumentParser(description='Evaluation of Part Localization Error')
parser.add_argument('--load', default='', type=str, help='name of model to evaluate')
args = parser.parse_args()

def calc_center(assignment, num_parts):
    """
    Calculate the geometric centers for assignment maps.

    Parameters
    ----------
    assignment: torch.cuda.FloatTensor, [batch_size, num_parts, 16, 16]
        Soft assignment map before upsampling.
    num_parts: int, number of object parts.

    Returns
    ----------
    centers: torch.cuda.FloatTensor, [batch_size, num_parts * 2]
        Geometric centers for assignment maps with order
        (col_1, row_1, ..., col_K, row_K)

    """
    # row and col check, x means row num here
    batch_size = assignment.shape[0]
    nparts = assignment.shape[1]
    height = assignment.shape[2]
    width = assignment.shape[3]

    # assertions
    assert nparts == num_parts
    assert height == 16
    assert width == 16

    # generate the location map
    col_map = torch.arange(1, width+1).view(1, 1, 1, width).expand(batch_size, nparts, height, width).float().cuda()
    row_map = torch.arange(1, height+1).view(1, 1, height, 1).expand(batch_size, nparts, height, width).float().cuda()

    # multiply the location map with the soft assignment map
    col_weighted = (col_map * assignment).view(batch_size, nparts, -1)
    row_weighted = (row_map * assignment).view(batch_size, nparts, -1)

    # sum of assignment as the denominator
    denominator = torch.sum(assignment.view(batch_size, nparts, -1), dim=2) + 1e-8

    # calculate the weighted average of location maps as centers
    col_mean = torch.sum(col_weighted, dim=2) / denominator
    row_mean = torch.sum(row_weighted, dim=2) / denominator

    # prepare the centers for return
    col_centers = col_mean.unsqueeze(2)
    row_centers = row_mean.unsqueeze(2)

    # N * K * 1 -> N * K * 2 -> N * (K * 2)
    centers = torch.cat([col_centers, row_centers], dim=2).view(batch_size, nparts * 2)

    # upsample to the image resolution (256, 256)
    centers = centers * 16

    return centers

def create_centers(data_loader, model, num_parts):
    """
    Generate the center coordinates as tensor for the current model and data split.
    Please make sure you are evaluating the model trained under 'interpretability' split.

    Parameters
    ----------
    data_loader: torch.utils.data.DataLoader
        Data loader for the current data split.
    model: pytorch model object
        Model that generates assignment maps.
    num_parts: int, number of object parts

    Returns
    ----------
    centers_tensor: torch.cuda.FloatTensor, [data_size, num_parts * 2]
        Geometric centers for assignment maps of the whole dataset.
        The coordinate order is (col_1, row_1, ..., col_K, row_K)
    annos_tensor: torch.cuda.FloatTensor, [batch_size, num_landmarks * 2]
        Landmark coordinate annotations of the whole dataset.
        The coordinate order is (col_1, row_1, ..., col_K, row_K)
    eyedists_tensor: torch.cuda.FloatTensor, [batch_size, 1]
        Eye (inter-ocular) distances of the whole dataset.

    """
    # tensor for collecting centers, labels and normalization terms
    centers_collection = []
    annos_collection = []
    eyedists_collection = []

    # iterating the data loader, landmarks shape: [N, num_landmarks, 2], column first
    for i, (input, _, landmarks) in enumerate(data_loader):

        # to device
        input = input.cuda()
        landmarks = landmarks.cuda()

        # gather the landmark annotations and the center outputs
        with torch.no_grad():

            # generate assignment map
            _, _, assignment = model(input)

            # calculate the center coordinates of shape [N, K * 2]
            centers = calc_center(assignment, num_parts)

            # collect the centers and annotations
            centers_collection.append(centers)
            annos_collection.append(landmarks)

            # calculate and collect the normalization factor
            col_dist = landmarks[:, 1, 0] - landmarks[:, 0, 0]
            row_dist = landmarks[:, 1, 1] - landmarks[:, 0, 1]
            eye_dist = (col_dist * col_dist + row_dist * row_dist).sqrt()
            eyedists_collection.append(eye_dist)

    # list into tensors
    centers_tensor = torch.cat(centers_collection, dim=0)
    annos_tensor = torch.cat(annos_collection, dim=0)
    eyedists_tensor = torch.cat(eyedists_collection, dim=0)

    # reshape the annotation and normalization tensor
    annos_tensor = annos_tensor.contiguous().view(centers_tensor.shape[0], num_landmarks * 2)
    eyedists_tensor = eyedists_tensor.contiguous().unsqueeze(1)

    return centers_tensor, annos_tensor, eyedists_tensor

def L2_distance(prediction, annotation):
    """
    Average L2 distance of two numpy arrays.

    Parameters
    ----------
    prediction: np.array, [data_size, num_landmarks, 2]
        Landmark prediction.
    annotation: np.array, [data_size, num_landmarks, 2]
        Landmark annotation.

    Returns
    ----------
    error: float
        Average L2 distance between prediction and annotation.

    """
    diff_sq = (prediction - annotation) * (prediction - annotation)
    L2_dists = np.sqrt(np.sum(diff_sq, axis=2))
    error = np.mean(L2_dists)
    return error

def main():

    # load the config file
    config_file = '../../log/'+ args.load +'/train_config.json'
    with open(config_file) as fi:
        config = json.load(fi)
        print(" ".join("\033[96m{}\033[0m: {},".format(k, v) for k, v in config.items()))

    # test transform
    data_transforms = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                            std=(0.229, 0.224, 0.225))
        ])

    # define dataset and loader
    assert config['split'] == 'interpretability'
    fit_data = CelebA('../../data/celeba',
        split='fit', align=False, percentage=0.3, transform=data_transforms, resize=(256, 256))
    eval_data = CelebA('../../data/celeba',
        split='eval', align=False, percentage=0.3, transform=data_transforms, resize=(256, 256))
    fit_loader = torch.utils.data.DataLoader(
        fit_data, batch_size=config['batch_size'], shuffle=False,
        num_workers=config['workers'], pin_memory=False, drop_last=False)
    eval_loader = torch.utils.data.DataLoader(
        eval_data, batch_size=config['batch_size'], shuffle=False,
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

    # convert the assignment to centers for both splits
    print('Evaluating the model for the whole data split...')
    fit_centers, fit_annos, fit_eyedists = create_centers(
        fit_loader, model, config['nparts'])
    eval_centers, eval_annos, eval_eyedists = create_centers(
        eval_loader, model, config['nparts'])
    eval_data_size = eval_centers.shape[0]

    # normalize the centers to make sure every face image has unit eye distance
    fit_centers, fit_annos = fit_centers / fit_eyedists, fit_annos / fit_eyedists
    eval_centers, eval_annos = eval_centers / eval_eyedists, eval_annos / eval_eyedists

    # fit the linear regressor with sklearn
    # normalized assignment center coordinates -> normalized landmark coordinate annotations
    print('=> fitting and evaluating the regressor')

    # convert tensors to numpy (64 bit double)
    fit_centers_np = fit_centers.cpu().numpy().astype(np.float64)
    fit_annos_np = fit_annos.cpu().numpy().astype(np.float64)
    eval_centers_np = eval_centers.cpu().numpy().astype(np.float64)
    eval_annos_np = eval_annos.cpu().numpy().astype(np.float64)

    # data standardization
    scaler_centers = StandardScaler()
    scaler_landmarks = StandardScaler()

    # fit the StandardScaler with the fitting split
    scaler_centers.fit(fit_centers_np)
    scaler_landmarks.fit(fit_annos_np)

    # stardardize the fitting split
    fit_centers_std = scaler_centers.transform(fit_centers_np)
    fit_annos_std = scaler_landmarks.transform(fit_annos_np)

    # define regressor without intercept and fit it
    regressor = LinearRegression(fit_intercept=False)
    regressor.fit(fit_centers_std, fit_annos_std)

    # standardize the centers on the evaluation split
    eval_centers_std = scaler_centers.transform(eval_centers_np)

    # regress the landmarks on the evaluation split
    eval_pred_std = regressor.predict(eval_centers_std)

    # unstandardize the prediction with StandardScaler for landmarks
    eval_pred = scaler_landmarks.inverse_transform(eval_pred_std)

    # calculate the error
    eval_pred = eval_pred.reshape((eval_data_size, num_landmarks, 2))
    eval_annos = eval_annos_np.reshape((eval_data_size, num_landmarks, 2))
    error = L2_distance(eval_pred, eval_annos) * 100

    print('Mean L2 Distance on the test set is %.2f%%.' % error)
    print('Evaluation finished for model \''+args.load+'\'.')

if __name__ == '__main__':
    main()
