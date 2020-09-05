""" Class that download weights, retrain a CNN model and predict people emotions and gender.

Adapted from:
    https://github.com/mdietrichstein/tensorflow-open_nsfw

Author:
    Ahmed Haj Yahmed (hajyahmedahmed@gmail.com)
"""
# import tensorflow as tf
import numpy as np
import tensorflow.compat.v1 as tf
import cv2
import os
import csv
import pandas as pd
from nudity_detection.nsfw_detector.model import OpenNsfwModel, InputType
from nudity_detection.nsfw_detector.image_utils import create_tensorflow_image_loader
from nudity_detection.nsfw_detector.image_utils import create_yahoo_image_loader

tf.disable_v2_behavior()

PATH = 'C:/Users/ASUS/Desktop/stage/nudity_detection/packaging/'


class NsfwDetector:
    """A class that contains all steps to handel a CNN model for NSFW detection.

    This Class download the pre-trained weights, load the model, eventually re-train the model
    and detect giving a picture.

    Attributes:
        IMAGE_LOADER_TENSORFLOW (str): tensorflow image loader label.
        IMAGE_LOADER_YAHOO (str): yahoo image loader label.
        model (tensorflow.model): tensorflow model.
        labels (list): training labels.
        sess (Tensorflow.Session): running session.
        fn_load_image : image loader.
    """

    def __init__(self,
                 weight_file_path=PATH + 'nudity_detection/nudity_detection/nsfw_detector/data/open_nsfw-weights.npy',
                 input_type=InputType.TENSOR.name.lower(),
                 image_loader="yahoo"):
        """load the NFSW detection model, run the session
        and define the list of labels.

        Args:
            weight_file_path (str): path of the CNN model weights.
        """
        self.IMAGE_LOADER_TENSORFLOW = "tensorflow"
        self.IMAGE_LOADER_YAHOO = "yahoo"
        self.model = OpenNsfwModel()
        self.labels = ['SFW', 'NSFW']
        self.sess = tf.Session()
        input_type = InputType[input_type.upper()]
        self.model.build(weights_path=weight_file_path, input_type=input_type)

        self.fn_load_image = None
        if input_type == InputType.TENSOR:
            if image_loader == self.IMAGE_LOADER_TENSORFLOW:
                self.fn_load_image = create_tensorflow_image_loader(tf.Session(graph=tf.Graph()))
            else:
                self.fn_load_image = create_yahoo_image_loader()
        elif input_type == InputType.BASE64_JPEG:
            import base64
            self.fn_load_image = lambda filename: np.array([base64.urlsafe_b64encode(open(filename, "rb").read())])

        self.sess.run(tf.global_variables_initializer())

    def download(self):
        """download the CNN model weights and the haarcascade xml file if not found.
        """
        pass

    def predict(self, image_path,
                results_dir=PATH + 'nudity_detection/nudity_detection/nsfw_detector/results'):
        """detect NSFW objects given an image.

        detect NSFW in a picture. This method create an output folder containing
        the input image with the bounding box and the class of each face detected and a xlsx file.

        args:
            image_path (str): path of the input image.
            results_dir (str): path of the output folder containing the output image and the output xlsx file.

        returns:
            read_file (pandas.Dataframe) : dataframe containing prediction results.
        """
        # create the result folder if not exist
        if not (os.path.isdir(results_dir)):
            os.mkdir(results_dir)
        # create the csv file
        base = os.path.basename(image_path)
        file_name = os.path.splitext(base)[0]
        with open(results_dir + '/out_' + file_name + '.csv', 'w', newline='') as file:
            writer = csv.writer(file, dialect='excel', quotechar='"', quoting=csv.QUOTE_ALL, delimiter=',')
            writer.writerow(["class_nudity_detector", "probability_nudity_detector"])

        # Only jpeg, jpg images are supported
        img = cv2.imread(image_path)
        image = self.fn_load_image(image_path)

        predictions = self.sess.run(self.model.predictions, feed_dict={self.model.input: image})

        # print("Results for '{}'".format(image_path))
        # print("\tSFW score:\t{}\n\tNSFW score:\t{}".format(*predictions[0]))
        predicted_class = self.labels[np.argmax(predictions[0])]
        predicted_prob = np.amax(predictions[0])
        with open(results_dir + '/out_' + file_name + '.csv', 'a', newline='') as file:
            writer = csv.writer(file, dialect='excel', quotechar='"', quoting=csv.QUOTE_ALL, delimiter=',')
            writer.writerow([predicted_class, predicted_prob])
        cv2.putText(img, predicted_class, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 50, 155), 2)
        cv2.imwrite(results_dir + '/out_' + base, img)
        read_file = pd.read_csv(results_dir + '/out_' + file_name + '.csv')
        read_file.to_excel(results_dir + '/out_' + file_name + '.xlsx', index=None, header=True)
        os.remove(results_dir + '/out_' + file_name + '.csv')
        return read_file

    def train(self):
        """re-trained the model for better accuracy.
        """
        pass
