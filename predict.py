import argparse
import numpy as np
from tensorflow.keras.models import load_model

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
