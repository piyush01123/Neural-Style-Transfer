
from tensorflow.keras.applications import vgg19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

import scipy.misc
import numpy as np

VGG = vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling=None)

IMAGE_WIDTH = 400
IMAGE_HEIGHT = 300
COLOR_CHANNELS = 3
NOISE_RATIO = 0.6
MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
OUTPUT_DIR = 'output/'
ALPHA = 10 #contribution of content cost to total cost
BETA = 40  #contribution of style cost to total cost
LAYER_WEIGHTS = [('block1_conv1', 0.2), ('block2_conv1', 0.2), ('block3_conv1', 0.2), ('block4_conv1', 0.2), ('conv5_1', 0.2)]
CONTENT_LAYER = 'block4_conv2'
LR = .001

class GatysNeuralStyle:
    def __init__(self, content_img_path, style_img_path, output_dir=OUTPUT_DIR, \
                 image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT, layer_weights=LAYER_WEIGHTS, \
                 noise_ratio=NOISE_RATIO, means= MEANS, alpha=ALPHA, beta=BETA, content_layer=CONTENT_LAYER,\
                 lr=LR):

        self.image_width, self.image_height = image_width, image_height
        self.means = means
        self.noise_img = np.random.uniform(-20, 20, (1, self.image_width, self.image_height, 3)).astype('float32')
        self.content_img = self.load_img(content_img_path)
        self.style_img = self.load_img(style_img_path)
        self.output_dir = output_dir
        self.generated_img = noise_ratio*self.noise_img + (1-noise_ratio)*self.content_img

        self.layer_weights = layer_weights
        self.alpha = alpha
        self.beta = beta
        self.content_layer = content_layer
        self.lr = lr
        print('INIT??')

    def compute_content_cost(self):
        self.content_model = Model(inputs=VGG.inputs, outputs=VGG.get_layer(self.content_layer).output)
        self.a_C = self.content_model(input_tensor=self.content_img)
        self.a_G = self.content_model(input_tensor=self.generated_img)
        self.a_C = Flatten()(self.a_C)
        self.a_G = Flatten()(self.a_G)
        return mean_squared_error(self.a_C, self.a_G)

    @staticmethod
    def gram_matrix(A):
        return tf.matmul(A, tf.transpose(A))

    def compute_layer_style_cost(self, layer):
        self.layer_model = Model(inputs=VGG.inputs, outputs=VGG.get_layer(layer).output)
        self.a_S = self.layer_model(input_tensor=self.style_img)
        self.a_G = self.layer_model(input_tensor=self.generated_img)
        self.a_S = self.gram_matrix(self.a_S)
        self.a_G = self.gram_matrix(self.a_G)
        return mean_squared_error(self.a_S, self.a_G)

    def compute_style_cost(self):
        self.style_cost = 0
        for layer, weight in self.layer_weights:
            self.style_cost += weight*self.compute_style_cost(layer)
        return self.style_cost

    def compute_total_cost(self):
        self.content_cost = self.compute_content_cost()
        self.style_cost = self.compute_layer_style_cost()
        return self.alpha*self.content_cost + self.beta*self.style_cost

    def minimize_cost(self):
        self.total_cost = self.compute_total_cost()
        opt = Adam(lr=self.lr)
        return opt.minimize(self.total_cost)

    def stylize(self):
        for step in range(100):
            self.minimize_cost()
            if step % 10==0:
                self.save_img('self.output_dir_step%s.jpg' %step, self.generated_img)

    def load_img(self, path):
        img = scipy.misc.imread(path, mode='RGB').astype(np.float32)
        img = scipy.misc.imresize(img, (self.image_width, self.image_height))
        img = np.reshape(img, (1, self.image_width, self.image_height, 3))
        img = img - self.means
        return img

    def save_img(self,  path, img):
        img = img + self.means
        img = np.clip(img[0], 0, 255).astype('uint8')
        scipy.misc.imsave(path, image)

if __name__=='__main__':
    stylizer = GatysNeuralStyle(content_img_path='piyush.png', style_img_path='van_gogh.jpg')
    stylizer.stylize()
