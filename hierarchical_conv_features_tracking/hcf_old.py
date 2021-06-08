import cv2
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from numpy import conj, real
import argparse
import cv2
import os
from os.path import join, realpath, dirname
from keras.applications.vgg19 import VGG19
from keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims
from os.path import realpath, dirname, join
import cv2
import scipy
from utils.utils import gaussian2d_rolled_labels, cos_window
import scipy.io

style_layers = ['block5_conv4',
                'block4_conv4',
                'block3_conv4']




def create_model2():
    import tensorflow as tf
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    indLayers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                 20, 21]  # VGG19 3., 4., 5. cnn layers  [20, 15, 10]

    #outputs = [vgg.get_layer(name).output for name in style_layers]
    outputs = [vgg.layers[i].output for i in indLayers]

    model = tf.keras.Model([vgg.input], outputs)
    return model

def create_model1():
    mat = scipy.io.loadmat('D:/PROJECTS/MOT_CNN_DATAASSOCIATION/imagenet-vgg-verydeep-19.mat')
    model = VGG19(mat)
    nweights = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 1,
                0]  # Weights for combining correlation filter responses [1, 0.5, 0.02]
    indLayers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                 20, 21]  # VGG19 3., 4., 5. cnn layers  [20, 15, 10]

    outputs = [model.layers[i].output for i in indLayers]

    return Model(inputs=model.inputs, outputs=outputs)

class HCFTracker():
    def __init__(self):
        self.model = create_model2()
        self.indLayers = [20, 15, 10]  # The CNN layers Conv5-4, Conv4-4, and Conv3-4 in VGG Net
        self.nweights = [1, 0.5, 0.02]  # Weights for combining correlation filter responses

        self.padding = {"generic": 1.8, "large": 1, "height": 0.4} # generic, large, height
        self.sigma = 0.6
        self.lambdar = 0.0001 # Regularization parameter Eqn 3
        self.output_sigma_factor = 0.1  # Spatial bandwidth (proportional to the target size)
        self.interp_factor = 0.01 # Model learning rate (see Eqn 6a, 6b)
        self.cell_size = 4 # Spatial cell size
        self.numLayers = len(self.indLayers)
        self.model_xf = []
        self.model_x = []
        self.model_alphaf = []

    def init(self, image, roi):
        # Get image size and search window size
        x1, y1, w, h = roi
        self.crop_size = (w, h)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.target_sz = [h, w] # 125 122
        im_sz = image.shape
        self.window_sz = self.get_search_window(self.target_sz, im_sz, self.padding)
        #  Compute the sigma for the Gaussian function label
        output_sigma = np.sqrt(self.target_sz[0] * self.target_sz[1]) * self.output_sigma_factor / self.cell_size # 3.08
        # create regression labels, gaussian shaped, with a bandwidth
        self.l1_patch_num = np.floor(self.window_sz / self.cell_size) # 62 61
        # Pre-compute the Fourier Transform of the Gaussian function label
        self.yf = fft2(gaussian2d_rolled_labels([self.l1_patch_num[1], self.l1_patch_num[0]], output_sigma)) # 62 61 complex double
        # Pre-compute and cache the cosine window (for avoiding boundary discontinuity)
        self.cos_window = cos_window([self.yf.shape[1], self.yf.shape[0]]) # 62 61 double
        _nweights = np.zeros([1, 1, 3]) # ELNURA
        _nweights[0, 0, 0] = self.nweights[0]
        _nweights[0, 0, 1] = self.nweights[1]
        _nweights[0, 0, 2] = self.nweights[2] # ELNURA
        self.nweights = _nweights # 1x1x3 ELNURA Düzenle
        # Note: variables ending with 'f' are in the Fourier domain.
        self.model_xf = [] # numLayers
        self.model_x = []
        self.model_alphaf = [] # numLayers
        self.current_scale_factor = 1

        self.pos = [roi[1] + np.floor(self.target_sz[0] / 2), roi[0] + np.floor(self.target_sz[1] / 2)]  # 130, 178

        # Extracting hierarchical convolutional features
        feat = self.extract_feature(image, self.pos, self.window_sz, self.cos_window, self.indLayers)
        self.update_model(feat, self.yf, self.interp_factor, self.lambdar, 0)
        target_sz_t = self.target_sz * self.current_scale_factor
        box = [self.pos[1] - target_sz_t[1] / 2,
               self.pos[0] - target_sz_t[0] / 2,
               target_sz_t[1],
               target_sz_t[0]]  # 117, 67.5, 122, 125

        return box[0], box[1], box[2], box[3]

    def update(self, image, vis=False):
        if vis is True:
            self.score = self.yf
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Extracting hierarchical convolutional features
        feat = self.extract_feature(image, self.pos, self.window_sz, self.cos_window, self.indLayers)
        # Predict position
        pos = self.predict_position(feat, self.pos, self.indLayers, self.nweights, self.cell_size,
                                       self.l1_patch_num)

        # Extracting hierarchical convolutional features
        feat = self.extract_feature(image, pos, self.window_sz, self.cos_window, self.indLayers)
        self.update_model(feat, self.yf, self.interp_factor, self.lambdar, 1)
        target_sz_t = self.target_sz * self.current_scale_factor
        self.pos = pos
        box = [pos[1] - target_sz_t[1] / 2,
               pos[0] - target_sz_t[0] / 2,
               target_sz_t[1],
               target_sz_t[0]]  # 117, 67.5, 122, 125

        return box[0], box[1], box[2], box[3]

    def extract_feature(self, im, pos, window_sz, cos_window, indLayers):
        # Get the search window from previous detection
        patch = self.get_subwindow(im, pos, window_sz)
        #patch = self._preprocessing(patch, cos_window) # ELNURA
        features = self.get_features(patch, cos_window, indLayers)

        return features

    def gaussian_peak(self, w, h):
        output_sigma = 0.125
        sigma = np.sqrt(w * h) / self.padding * output_sigma
        syh, sxh = h // 2, w // 2
        y, x = np.mgrid[-syh:-syh + h, -sxh:-sxh + w]
        x = x + (1 - w % 2) / 2.
        y = y + (1 - h % 2) / 2.
        g = 1. / (2. * np.pi * sigma ** 2) * np.exp(-((x ** 2 + y ** 2) / (2. * sigma ** 2)))
        return g

    def get_search_window(self, target_sz, im_sz, padding):
        if target_sz[0] / target_sz[1] > 2:
            # For objects with large height, we restrict the search window with padding.height
            # Element-wise multiplication
            window_sz = np.floor(target_sz * [1 + padding["height"], 1 + padding["generic"]]) # ELNURA
        elif (target_sz[0] * target_sz[1]) / (im_sz[0] * im_sz[1]) > 0.05:
            # 125 * 122 / 360 * 640
            # For objects with large height and width and accounting for at least 10 percent of the whole image,
            # we only search 2x height and width
            window_sz = np.floor(np.multiply(target_sz, (1 + padding["large"])))
        else:
            # otherwise, we use the padding configuration
            window_sz = np.floor(np.multiply(target_sz, (1 + padding["generic"])))

        return window_sz

    def get_subwindow(self, im, pos, sz):
        _p1 = np.array(range(1, int(sz[0] + 1))).reshape([1, int(sz[0])])
        #_p1 = _p1.reshape([_p1.shape[1], _p1.shape[0]])
        _p2 = np.array(range(1, int(sz[1] + 1))).reshape([1, int(sz[1])])
        #_p2 = _p2.reshape([_p2.shape[1], _p2.shape[0]])
        ys = np.floor(pos[0]) + _p1 - np.floor(sz[0]/2) # 1 250
        xs = np.floor(pos[1]) + _p2 - np.floor(sz[1]/2) # 1, 244
        #sub = cv2.getRectSubPix(im, sz, pos)
        # sub_image = im[y:y + h, x:x + w, :]
        # Check for out-of-bounds coordinates, and set them to the values at the borders
        xs = self.clamp(xs, 1, np.size(im, 1))
        ys = self.clamp(ys, 1, np.size(im, 0))
        # extract image
        out = im[int(ys[0][0]) - 1:int(ys[0][np.size(ys, 1)-1]),
              int(xs[0][0]) - 1: int(xs[0][np.size(xs, 1)-1]), :]

        return out

    def clamp(self, x, lb, ub):
        # Clamp the value using lowerBound and upperBound
        lb = np.full((x.shape[0], x.shape[1]), lb)
        ub = np.full((x.shape[0], x.shape[1]), ub)
        y = np.maximum(x, lb)
        y = np.minimum(y, ub)

        return y

    def get_features(self, im, cos_window, layers):
        sz_window = np.array([cos_window.shape[0], cos_window.shape[1]]).reshape([1, int(2)])

        # Preprocessing
        img = im.astype('float64')  # note: [0, 255] range
        img = cv2.resize(img, (224, 224))

        average = np.zeros((224, 224, 3)).astype('float64')
        average[:, :, 0] = 123.6800    # RGB 123, BGR 103
        average[:, :, 1] = 116.7790
        average[:, :, 2] = 103.9390

        img = np.subtract(img, average)
        img = img.round(4)

        img = expand_dims(img, axis=0)
        #img = preprocess_input(img)

        feature_maps = self.model.predict(img)
        features = []

        for fi in self.indLayers:
            f_map = feature_maps[fi][0][:][:][:]
            feature_map_n = cv2.resize(f_map, (cos_window.shape[1], cos_window.shape[0]), interpolation=cv2.INTER_LINEAR)
            feature_map_c = feature_map_n * cos_window[:, :, None]
            features.append(feature_map_c)

        return features

    def update_model(self, feat, yf, interp_factor, lambdar, frame):
        numLayers = len(feat)
        self.model_x = feat
        xf = []
        alphaf = []
        # Model update
        for i in range(numLayers):
            xf.append(fft2(feat[i]))
            kf = np.sum(conj(xf[i]) * xf[i], 2) / np.size(xf[i])
            alphaf.append(yf / (kf + lambdar)) #  Fast training
        # Model initialization or update
        if frame == 0: #  First frame, train with a single image
            self.model_alphaf = alphaf
            self.model_xf = xf
        else:
            for i in range(numLayers):
                self.model_alphaf[i] = (1 - interp_factor) * self.model_alphaf[i] + interp_factor * alphaf[i]
                self.model_xf[i] = (1 - interp_factor) * self.model_xf[i] + interp_factor * xf[i]

    def predict_position(self, feat, pos, indLayers, nweights, cell_size, l1_patch_num):
        # Compute correlation filter responses at each layer
        res_layer = np.zeros([int(l1_patch_num[0]), int(l1_patch_num[1]), len(indLayers)])

        for i in range(len(indLayers)):
            zf = fft2(feat[i])
            zf_c = zf * conj(self.model_xf[i])
            kzf = (np.sum(zf_c, axis=2)) / (np.size(zf)) # 62 61
            # kzf=sum(zf .* conj(model_xf{ii}), 3) / numel(zf)
            # ifft2(np.sum(conj(fft2(x1)) * fft2(x2), axis=0))
            # real(ifft2(alphaf * fft2(k)))
            temp = real(fftshift(ifft2(self.model_alphaf[i] * kzf))) # equation for fast detection 62 61, nweights -> 1 1 3
            #  temp= real(fftshift(ifft2(model_alphaf{ii} .* kzf)));  %equation for fast detection
            res_layer[:, :, i] = temp / np.max(temp)

        # Combine responses from multiple layers (see Eqn. 5)
        # response = sum(bsxfun(@times, res_layer, nweights), 3)
        response = np.sum(res_layer * nweights, axis=2)

        # Find target location
        # if the target doesn't move, the peak will appear at the top-left corner, not at the center.
        # The responses wrap around cyclically.

        # Find indices and values of nonzero elements curr = np.unravel_index(np.argmax(gi, axis=None), gi.shape)
        delta = np.unravel_index(np.argmax(response, axis=None), response.shape)
        vert_delta, horiz_delta = delta[0], delta[1]
        vert_delta = vert_delta - np.floor(np.size(zf, 0) / 2)
        horiz_delta = horiz_delta - np.floor(np.size(zf, 1) / 2)

        # Map the position to the image space
        pos = np.array(pos)
        pos = pos.reshape((pos.shape[0], 1))
        move_pos = np.array([vert_delta, horiz_delta]).reshape((pos.shape[0], 1))
        pos = pos + self.cell_size * move_pos# 122 174

        return pos

    def _preprocessing(self, img, cos_window, eps=1e-5):
        img = np.log(img+1)
        img = (img-np.mean(img))/(np.std(img)+eps)
        return cos_window*img

def get_img_list(img_dir):
    frame_list = []
    for frame in sorted(os.listdir(img_dir)):
        if os.path.splitext(frame)[1] == '.jpg':
            frame_list.append(os.path.join(img_dir, frame))
    return frame_list

'''
if __name__ == '__main__':
    tracker = HCFTracker()
    roi = (275,	137, 23, 26) # surfer (275,	137, 23, 26)  (117, 68, 122, 125)
    img_dir = 'D:/PROJECTS/DATASET/OTB50/Surfer/img/'
    frame_list = get_img_list(img_dir)
    frame_list.sort()
    current_frame = cv2.imread(frame_list[0])
    tracker.init(current_frame, roi)

    for idx in range(len(frame_list)):
        frame = cv2.imread(frame_list[idx])
        bbox = tracker.update(frame)
        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
        cv2.imwrite('D:/PROJECTS/DATASET/OTB50/Surfer/img/RESULT/HCF/' + str(idx) + '.jpg', frame)
        cv2.imshow('tracking_single', frame)
        c = cv2.waitKey(1) & 0xFF

        if c == 27 or c == ord('q'):
            break

    cv2.destroyAllWindows()
'''

