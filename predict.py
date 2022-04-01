import argparse
import numpy as np
import cv2
from tensorflow.keras.models import load_model

def get_pad_len(array, from_back=False):
    # Count padding pixels.
    pad_val = 0
    pad_len = 0
    if from_back:
        array = np.flip(array)
    for val in array:
        if val != pad_val:
            break
        pad_len += 1
    return pad_len

def rm_padding(np_img):
    #np_img = np_img.squeeze(-1)
    h_img = np_img.shape[0]
    w_img = np_img.shape[1]
    sum_across_w = np_img.sum(axis=1) # summation across width
    sum_across_h = np_img.sum(axis=0) # summation across height

    pad_top = get_pad_len(sum_across_w)
    pad_bottom = get_pad_len(sum_across_w, from_back=True)
    pad_left = get_pad_len(sum_across_h)
    pad_right = get_pad_len(sum_across_h, from_back=True)

    np_img = np_img[pad_top:(h_img-pad_bottom),pad_left:(w_img-pad_right)]

    h_img_cropped = np_img.shape[0]
    w_img_cropped = np_img.shape[1]

    if h_img_cropped > w_img_cropped:
        pad = h_img_cropped - w_img_cropped
        pad_L = pad // 2
        pad_R = pad_L + (pad % 2)
        np_img = np.pad(np_img, pad_width=((0,0),(pad_L,pad_R)), mode='constant')
    else:
        pad = w_img_cropped - h_img_cropped
        pad_T = pad // 2
        pad_B = pad_T + (pad % 2)
        np_img = np.pad(np_img, pad_width=((pad_T,pad_B),(0,0)), mode='constant')

    return np_img

def img2np(img_path, pad_width=4, h_img=28, w_img=28, rm_pad=True):
    #h_img = w_img = 28
    x = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    #compute a bit-wise inversion so black becomes white and vice versa
    #x = np.invert(x)
    # Delete padding
    if rm_pad:
        x = rm_padding(x)
    #make it the right size
    #x = cv2.resize(x,(h_img, w_img))
    x = cv2.resize(x,(h_img-2*pad_width, w_img-2*pad_width))
    # Add padding
    x = np.pad(x, pad_width, 'constant')
    #x = imresize(x,(28,28))
    #convert to a 4D tensor to feed into our model
    x = x.reshape(1,h_img,w_img,1)
    x = x.astype('float32')
    x /= 255

    return x

def predict(img_path, model_path):
    np_img = img2np(img_path)
    #perform the prediction
    model = load_model(model_path)
    out = model.predict(np_img)
    pred = np.argmax(out)

    return pred

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Predict the class of a given digit image.')
    parser.add_argument('--img_path', type=str, required=True,
        help='The path of an image to be predicted')
    parser.add_argument('--model_path', type=str, default='cnn.h5',
        help='The path of a model to predict a given image')
    args = parser.parse_args()

    args.img_path
    args.model_path

    pred = predict(args.img_path, args.model_path)

    print('Prediction:', pred)
