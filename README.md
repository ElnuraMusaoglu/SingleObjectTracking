# SingleObjectTracking
Single object tracking using OTB-100 dataset
Tracking methods : ['MOSSE', 'Kalman',
'OPENCV_CSRT', 'OPENCV_GOTURN', 'OPENCV_BOOSTING', 'OPENCV_MEDIANFLOW', 'OPENCV_MIL', 'OPENCV_TLD', 'OPENCV_MOSSE', 
'KCF_GRAY', 'KCF_COLOR', 'KCF_HOG',
'HCF_C1', 'HCF_C2', 'HCF_C3', 'HCF_C4', 'HCF_C5', 'HCF']
 

References

[1] Y. Wu, J. Lim, M.-H. Yang, "Online Object Tracking: A Benchmark", CVPR 2013.
Website: http://visual-tracking.net/

[2] P. Dollar, "Piotr's Image and Video Matlab Toolbox (PMT)".
Website: http://vision.ucsd.edu/~pdollar/toolbox/doc/index.html

[3] https://github.com/lawpdas/fhog-python

[4] Bolme, D. S., Beveridge, J. R., Draper, B. A., Lui, Y. M. 2010. Visual object tracking using adaptive correlation filters. Computer VisIon and Pattern Recognition., pp. 2544–2550.

[5] Henriques, J.F., Caseiro, R., Martins, P. and Batista, J., 2014. Highspeed tracking with kernelized correlation filters. IEEE Transactions on Pattern Analysis and Machine Intelligence, 37(3), pp.583-596.

[6] Welch, G., Bishop, G. 1995. An introduction to the Kalman filter, University of North Carolina, Department of Computer Science, TR 95-041.

[7] Ma, C., Huang, C. J. B., Yang, X., Yang, M. H. 2015 .Hierarchical convolutional features for visual tracking. IEEE International Conference on Computer Vision (ICCV), Santiago, Chile.

# The steps to use Tracking Methods using HOG and VGG-19 deep features

## 1. Install and Build fhog function of PDollar Toolbox to the folder hog_cpp

python setup.py build_ext --inplace

## 2. Download the VGG-Net-19 model, replace to model folder
##    Set vgg_path in kernelized_correlation_filter.py line 15

http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat

## 3. Download OTB-100 Dataset to the dataset folder

http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html

# RESULTS for OTB-100 Dataset

![precision_result](https://user-images.githubusercontent.com/79086158/121249456-4657f900-c8ad-11eb-9a6e-c15b83a5d67d.png)
![precision_result2](https://user-images.githubusercontent.com/79086158/121249523-58399c00-c8ad-11eb-95b8-247cb5e18d91.png)
![precision_result3](https://user-images.githubusercontent.com/79086158/121249539-5d96e680-c8ad-11eb-8305-cfafbdd36aac.png)
![success_result](https://user-images.githubusercontent.com/79086158/121249542-5e2f7d00-c8ad-11eb-9b95-3d343841bc1b.png)
![success_result2](https://user-images.githubusercontent.com/79086158/121249543-5ec81380-c8ad-11eb-8f2e-a361ba3e62d2.png)
![success_result3](https://user-images.githubusercontent.com/79086158/121249545-5ec81380-c8ad-11eb-9f1e-3fa997751822.png)



