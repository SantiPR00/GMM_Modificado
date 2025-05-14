import cv2
import numpy as np
import math
from sklearn import mixture
from scipy import linalg
import itertools
import matplotlib.pyplot as plt
from scipy import linalg
import matplotlib as mpl
from time import time
from scipy import infty
from sklearn import preprocessing
from sklearn.utils import shuffle
from matplotlib import colors as mcolors
from skimage import color, data, restoration
from scipy.signal import convolve2d as conv2
from itertools import chain
from skimage import feature
from imageio import imread  # Reemplaza imread
from scipy.ndimage import convolve  # Reemplaza imfilter
from sklearn.preprocessing import StandardScaler
import os

color_iter = itertools.cycle(mpl.cm.get_cmap('tab10').colors)  # Colores más variados

color_code = {
    1: (255, 220, 60),   
    2: (180, 100, 190),  
    3: (130, 90, 130),   
    4: (70, 190, 220),  
    5: (255, 100, 255),  
    6: (210, 130, 40),   
    7: (100, 220, 90),  
}






def test(imagetest,gmm):
    pre = gmm.predict(imagetest)
    print(np.unique(pre))
    # plot_results(imagetest,pre,gmm.means_,gmm.covariances_,'Gaussian Mixture')
    # print gmm.means_
    # plt.show()

    return pre
def train(num_patches, img, n_samples, w, h):
    imtrain = shuffle(img)
    imtrain = imtrain[:200000]
    scaler = StandardScaler()
    imtrain_scaled = scaler.fit_transform(imtrain)  
    gmm = mixture.GaussianMixture(n_components=7, covariance_type='tied', 
                                  tol=1e-8, reg_covar=1e-6, max_iter=1200, n_init=1, 
                                  init_params='k-means++', warm_start=True, random_state=42).fit(imtrain)
    
    print("Converged:", gmm.converged_)  # Verificar si el GMM ha convergido
    
    return gmm


def segmented(image,samples,label,num_comp):
    labels = np.expand_dims(label, axis = 0)
    labels = np.transpose(labels)


    for i in range (1, num_comp):
        indices = np.where(np.all(labels == i, axis =-1))
        print(f"Píxeles en el componente {i}: {len(indices[0])}")  # Verifica la cantidad de píxeles asignados a cada clase
        indices = np.unravel_index(indices,(w,h), order= 'C')
        type(indices)
        indices = np.transpose(indices)

        l = chain.from_iterable(zip(*indices))

        for j, (lowercase, uppercase) in enumerate(l):
            # set the colour accordingly

            image[lowercase,uppercase] = color_code[(i)]
    return image

# local binary pattern descriptor
def createFeature(image, n_samples):
    numpoints = 24
    radius = 8
    img_src = cv2.GaussianBlur(image,(5,5),0)
    #img_src = cv2.bilateralFilter(image, 9, 75, 75)
    imtest = img_src
    #imtest = cv2.cvtColor(img_src,cv2.COLOR_BGR2LAB)

    # blur = cv2.bilateralFilter(img_src,9,75,75)
    # blurthresh=100
    # imtest = np.fix(imtest, blurthresh)
    img_gray = cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(img_gray,numpoints, radius, method="uniform")
    lbp = np.reshape(lbp,(n_samples,1))
    imtest = np.reshape(imtest,(n_samples,d))
    # print np.shape(lbp)
    data = np.column_stack((imtest, lbp))
    data = preprocessing.normalize(imtest, norm='l2')
    data= preprocessing.scale(data)
    data = preprocessing.minmax_scale(data, axis=0)


    return data, imtest

img = cv2.imread('2.png') #cambiado
gt_image = cv2.imread('2.png') #cambiado
b = gt_image[:,:,0] < 255
g = gt_image[:,:,1] 
r = gt_image[:,:,2] 
gt_image[b] = 0
gt_image[~b] = 1
img_src = cv2.multiply(gt_image,img)


w, h, d = tuple(img_src.shape)

# Number of samples per component
n_samples = w*h
#Number of sets of training samples
num_patches=100;

#print w,h

samples, imtest=createFeature(img_src, n_samples)
gmm = train(num_patches,samples,n_samples,w,h)
print (gmm.means_) #cambiado: ponemos paréntesis porque cambiamos de python2 a python3


image_test = cv2.imread('2.png') #cambiado
test_samples, im=createFeature(image_test, n_samples)
pre = test(test_samples,gmm)
#image_test = cv2.cvtColor(image_test, cv2.COLOR_BGR2LAB)
seg1 = segmented(image_test,test_samples,pre,7)
cv2.imwrite('segmentation.png', seg1)
k= cv2.waitKey(0)
if k ==27:
    cv2.destroyAllWindows()
