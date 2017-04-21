import os
import pathlib
import tarfile

import boto3
from botocore.handlers import disable_signing

from scipy.io import loadmat
from scipy.misc import imread

import numpy as np
from skimage.transform import resize


def list_files(base_path, validExts=(".jpg", ".jpeg", ".png", ".bmp"), contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(base_path):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if ext.endswith(validExts):
                # construct the path to the image and yield it
                imagePath = os.path.join(rootDir, filename).replace(" ", "\\ ")
                yield imagePath


class BSDS500(object):
    BUCKET = 'i008data'
    FN = 'BSR_bsds500.tgz'
    STORE_FN = 'BSR.tgz'

    def __init__(self, path_to_bsds=None, images_to_gray=False, target_size=None, masks_to_binary=True):
        if not path_to_bsds:
            self.BSDS_BASE = self.get_bsds()
        else:
            self.BSDS_BASE = path_to_bsds

        self.images_to_gray = images_to_gray
        self.target_size = target_size
        self.masks_to_binary = masks_to_binary

        self.TRAIN_PATH = os.path.join(self.BSDS_BASE, 'BSDS500/data/images/train/')
        self.TEST_PATH = os.path.join(self.BSDS_BASE, 'BSDS500/data/images/test/')
        self.VALID_PATH = os.path.join(self.BSDS_BASE, 'BSDS500/data/images/val/')
        self.GROUND_TRUTH_TRAIN = os.path.join(self.BSDS_BASE, 'BSDS500/data/groundTruth/train/')
        self.GROUND_TRUTH_TEST = os.path.join(self.BSDS_BASE, 'BSDS500/data/groundTruth/test/')
        self.GROUND_TRUTH_VALID = os.path.join(self.BSDS_BASE, 'BSDS500/data/groundTruth/val/')

    def get_bsds(self):
        if not pathlib.Path('BSR.tgz').exists():
            print("DOWNLOADING BSDS500 DATA BE PATIENT")
            s3_resource = boto3.resource('s3')
            s3_resource.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)
            bucket = s3_resource.Bucket(self.BUCKET)
            bucket.download_file(self.FN, self.STORE_FN)

        if not pathlib.Path('BSR').is_dir():
            tar = tarfile.open(self.STORE_FN)
            tar.extractall()

        dir_path = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(dir_path, self.STORE_FN.split('.')[0])

    def load_ground_truth(self, gt_path):
        ground_truth_paths = sorted(list(list_files(gt_path, validExts=('.mat'))))
        file_id = []
        cnts = []
        sgmnts = []

        for gt_path in ground_truth_paths:
            file_name = os.path.basename(gt_path).split('.')[0]
            gt = loadmat(gt_path)
            gt = gt['groundTruth'][0]
            for annotator in gt:
                contours = annotator[0][0][1]  # 1-> contours
                segments = annotator[0][0][0]  # 0 -> segments
                if self.target_size:
                    contours = resize(contours.astype(float), output_shape=self.target_size)
                    segments = resize(segments, output_shape=self.target_size)

                if self.masks_to_binary:
                    contours[contours > 0] = 1

                file_id.append(file_name)
                cnts.append(contours)
                sgmnts.append(segments)

        cnts = np.concatenate([np.expand_dims(a, 0) for a in cnts])
        sgmnts = np.concatenate([np.expand_dims(a, 0) for a in sgmnts])
        cnts = cnts[..., np.newaxis]
        sgmnts = sgmnts[..., np.newaxis]

        return file_id, cnts, sgmnts

    def load_images(self, list_of_files):
        processed_images = []
        for i, f in enumerate(list_of_files):

            if self.images_to_gray:
                im = imread(f, mode='L')
            else:
                im = imread(f)

            if self.target_size:
                im = resize(im, output_shape=self.target_size)

            processed_images.append(np.expand_dims(im, 0))

        processed_images = np.concatenate(processed_images)

        if self.images_to_gray:
            processed_images = processed_images[..., np.newaxis]

        return processed_images

    def get_train(self):
        file_ids, cnts, sgmnts = self.load_ground_truth(self.GROUND_TRUTH_TRAIN)
        image_paths = [self.TRAIN_PATH + f_id + '.jpg' for f_id in file_ids]
        images = self.load_images(image_paths)

        return file_ids, cnts, sgmnts, images

    def get_test(self):
        file_ids, cnts, sgmnts = self.load_ground_truth(self.GROUND_TRUTH_TEST)
        image_paths = [self.TEST_PATH + f_id + '.jpg' for f_id in file_ids]
        images = self.load_images(image_paths)

        return file_ids, cnts, sgmnts, images

    def get_val(self):
        file_ids, cnts, sgmnts = self.load_ground_truth(self.GROUND_TRUTH_VALID)
        image_paths = [self.VALID_PATH + f_id + '.jpg' for f_id in file_ids]
        images = self.load_images(image_paths)

        return file_ids, cnts, sgmnts, images

