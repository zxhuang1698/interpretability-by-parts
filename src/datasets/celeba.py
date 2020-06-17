import torch
import torch.utils.data as data
from torchvision import transforms

from PIL import Image
import os
import os.path
import pickle

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class CelebA(data.Dataset):
    """
    CelebA dataset.

    Variables
    ----------
        root, str: Root directory of the dataset.
        split, str: Current data split. 
            "train": Training split without MAFL images. (For localization)
            "train_full": Training split with MAFL images. (For classification)
            "val": Validation split for classification accuracy.
            "test": Testing split for classification accuracy.
            "fit": Split for fitting the linear regressor.
            "eval": Split for evaluating the linear regressor.
        align, bool: Whether use aligned version or not. 
        percentage, float: For unaligned version, the least percentage of (face area / image area)
        transform, callable: A function/transform that takes in a PIL.Image and transforms it.
        resize, tuple: The size of image (h, w) after transformation (This version does not support cropping)
    """
    def __init__(self, root, split='train', align=True, percentage=None, transform=None, resize=(256, 256)):

        self.root = root
        self.split = split
        self.align = align
        self.resize = resize

        # load the dictionary for data
        align_name = '_aligned' if align else '_unaligned'
        percentage_name = '_0' if percentage is None else '_'+str(int(percentage*100))
        save_name = os.path.join(root, split+align_name+percentage_name+'.pickle')

        if os.path.exists(save_name) is False:
            print('Preparing the data...')
            self.generate_dict(save_name)
            print('Data dictionary created and saved.')
        with open(save_name, 'rb') as handle:
            save_dict = pickle.load(handle)
        
        self.images = save_dict['images']           # image filenames
        self.landmarks = save_dict['landmarks']     # 5 face landmarks
        self.targets = save_dict['targets']         # binary labels
        self.bboxes = save_dict['bboxes']           # x y w h
        self.sizes = save_dict['sizes']             # height width
        self.transform = transform
        self.loader = pil_loader

        # select a subset of the current data split according the face area
        if percentage is not None:
            new_images = []
            new_landmarks = []
            new_targets = []
            new_bboxes = []
            new_sizes = []
            for i in range(len(self.images)):
                if float(self.bboxes[i][-1] * self.bboxes[i][-2]) >= float(self.sizes[i][-1] * self.sizes[i][-2]) * percentage:
                    new_images.append(self.images[i])
                    new_landmarks.append(self.landmarks[i])
                    new_targets.append(self.targets[i])
                    new_bboxes.append(self.bboxes[i])
                    new_sizes.append(self.sizes[i])
            self.images = new_images
            self.landmarks = new_landmarks
            self.targets = new_targets
            self.bboxes = new_bboxes                
            self.sizes = new_sizes                  

        print('Number of samples in the ' + self.split + ' split: '+ str(len(self.images)))


    # generate a dictionary for a certain data split
    def generate_dict(self, save_name):

        print('Start generating data dictionary as '+save_name)

        landmark = None
        full_img_list = []
        ann_file = 'list_attr_celeba.txt'
        bbox_file = 'list_bbox_celeba.txt'          
        size_file = 'list_imsize_celeba.txt'

        if self.align is True:
            landmark_file = 'list_landmarks_align_celeba.txt'
        else:
            landmark_file = 'list_landmarks_unalign_celeba.txt'

        # load all the images according to the current split
        if self.split == 'train':
            imgfile = 'celebA_training.txt'
        elif self.split == 'val':
            imgfile = 'celebA_validating.txt'
        elif self.split == 'test':
            imgfile = 'celebA_testing.txt'
        elif self.split == 'fit':
            imgfile = 'MAFL_training.txt'
        elif self.split == 'eval':
            imgfile = 'MAFL_testing.txt'
        elif self.split == 'train_full':
            imgfile = 'celebA_training_full.txt'
        for line in open(os.path.join(self.root, imgfile), 'r'):
            full_img_list.append(line.split()[0])

        # prepare the indexes and convert annotation files to lists
        full_img_list_idx = [(int(s.rstrip(".jpg"))-1) for s in full_img_list]
        ann_full_list = [line.split() for line in open(os.path.join(self.root, ann_file), 'r')]
        bbox_full_list = [line.split() for line in open(os.path.join(self.root, bbox_file), 'r')]
        size_full_list = [line.split() for line in open(os.path.join(self.root, size_file), 'r')]
        landmark_full_list = [line.split() for line in open(os.path.join(self.root, landmark_file), 'r')]

        # assertion
        assert len(ann_full_list[0]) == 41
        assert len(bbox_full_list[0]) == 5
        assert len(size_full_list[0]) == 3
        assert len(landmark_full_list[0]) == 11

        # select samples and annotations for the current data split
        # init the lists
        filename_list = []
        target_list = []
        landmark_list = []
        bbox_list = []
        size_list = []

        # select samples and annotations
        for idx in full_img_list_idx:

            # assertion
            assert (idx+1) == int(ann_full_list[idx][0].rstrip(".jpg"))
            assert (idx+1) == int(bbox_full_list[idx][0].rstrip(".jpg"))
            assert (idx+1) == int(size_full_list[idx][0].rstrip(".jpg"))
            assert (idx+1) == int(landmark_full_list[idx][0].rstrip(".jpg"))

            # append the filenames and annotations
            filename_list.append(ann_full_list[idx][0])
            target_list.append([int(i) for i in ann_full_list[idx][1:]])
            bbox_list.append([int(i) for i in bbox_full_list[idx][1:]])
            size_list.append([int(i) for i in size_full_list[idx][1:]])
            landmark_list_xy = []
            for j in range(5):
                landmark_list_xy.append([int(landmark_full_list[idx][1+2*j]), int(landmark_full_list[idx][2+2*j])])
            landmark_list.append(landmark_list_xy)

        # expand the filename to the full path
        if self.align is True:
            full_path_list = [os.path.join(self.root, 'aligned', filename) for filename in filename_list]
        else:
            full_path_list = [os.path.join(self.root, 'unaligned', filename) for filename in filename_list]
        
        # create the dictionary and save it on the disk
        save_dict = {}
        save_dict['images'] = full_path_list
        save_dict['landmarks'] = landmark_list
        save_dict['targets'] = target_list
        save_dict['bboxes'] = bbox_list
        save_dict['sizes'] = size_list
        with open(save_name, 'wb') as handle:
            pickle.dump(save_dict, handle)

    def __getitem__(self, index):
        """
        Retrieve data samples.

        Args
        ----------
        index: int
            Index of the sample.

        Returns
        ----------
        sample: PIL.Image
            Image of the given index.
        target: torch.LongTensor
            Binary labels for all attributes of the given index.
        landmark_locs: torch.FloatTensor, [5, 2]
            Landmark annotations, column first.
        """
        # load images and targets
        path = self.images[index]
        sample = self.loader(path)
        target = self.targets[index]
        target = torch.LongTensor(target)
        target = (target + 1) / 2
        width, height = sample.size

        # transform the image and target
        if self.transform is not None:
            sample = self.transform(sample)
        
        # processing the landmarks
        landmark_locs = self.landmarks[index]
        landmark_locs = torch.LongTensor(landmark_locs).float()
        landmark_locs[:, 0] = landmark_locs[:, 0] * self.resize[1] / float(width)
        landmark_locs[:, 1] = landmark_locs[:, 1] * self.resize[0] / float(height)

        return sample, target, landmark_locs

    def __len__(self):
        return len(self.images)

