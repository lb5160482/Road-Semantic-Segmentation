# Semantic Segmentation
[//]: # "Image References"
[image1]: ./output_img/000152_10.png	"Image_sample1"
[image2]: ./output_img/000164_10.png	"Image_sample2"
[image3]: ./output_img/000052_10.png	"Image_sample3"
[image4]: ./output_video/multi_class_seg_FPS20.gif	"gif1"
[image5]: ./readme_img/FCN_architecfture.png	"Architecture"
[image6]: ./readme_img/scaled_raw_image.png	"scaled"
[image7]: ./readme_img/flipped_raw_image.png	"flipped"
[image8]: ./readme_img/noise_raw_image.png	"nise"
[image9]: ./readme_img/lightened_raw_image.png	"darkened"



### Introduction

This project targets at labeling the pixels of an images(also knownas as Semantic Segmantation) using a Fully Convolutional Network (FCN). It implements transfer learning using VGG-16 and extract output of layer7, then used 1x1 convolution following by several transposed convolutional layer combine with skip-connection to upsample. The network architecture is like below:

![alt text][image5]

### Setup
##### GPU
Please make sure youTensorflow GPU is enabled. If you don't have a GPU on your system, you can use AWS or another cloud computing platform. This project uses [Floyd Hub](https://www.floydhub.com/).
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
### Dataset

Download the [Kitti Semantics Pixel-level data](http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015) from [here](http://www.cvlibs.net/download.php?file=data_semantics.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_semantics` with all the training a test images.

##### Label Format

name                     		  id      		category
'unlabeled'            		 ,  0 ,   		  'void'            
'ego vehicle'         		 ,  1 ,   		  'void'            
'rectification border' 	 ,  2 ,   		  'void'            
'out of roi'           		 ,  3 ,   		  'void'            
'static'              		 ,  4 ,    		  'void'            
'dynamic'             		 ,  5 ,    		  'void'            
'ground'              		 ,  6 ,    		  'void'            
'road'                		 ,  7 ,    		  'ground'          
'sidewalk'             		 ,  8 ,   		  'ground'          
'parking'             		 ,  9 ,    		  'ground'          
'rail track'           		 , 10 ,    		  'ground'          
'building'            		 , 11 ,   		  'construction'    
'wall'                 		 , 12 ,   		  'construction'    
'fence'                		 , 13 ,   		  'construction'    
'guard rail'         		 , 14 ,  		  'construction'    
'bridge'               		 , 15 ,   		  'construction'    
'tunnel'               		 , 16 ,    		  'construction'    
'pole'                 		 , 17 ,   		  'object'          
'polegroup'            	 , 18 ,   		  'object'          
'traffic light'        		 , 19 ,   		  'object'          
'traffic sign'         		 , 20 ,   		  'object'          
'vegetation'           	 , 21 ,   		  'nature'          
'terrain'             		 , 22 ,    		  'nature'          
'sky'                  		 , 23 ,    		  'sky'             
'person'              		 , 24 ,   		  'human'           
'rider'               		 , 25 ,    		  'human'           
'car'                 		 , 26 ,    		  'vehicle'        
'truck'               		 , 27 ,    		  'vehicle'        
'bus'                		 , 28 ,    		  'vehicle'        
'caravan'             		 , 29 ,    		  'vehicle'        
'trailer'            		 , 30 ,    		  'vehicle'        
'train'               		 , 31 ,    		  'vehicle'        
'motorcycle'           	 , 32 ,   		  'vehicle'        
'bicycle'              		 , 33 ,    		  'vehicle'        
'license plate'       		 , -1 ,    		  'vehicle'        

#### Data Augmentation
To train a more robust model, here I implemented data augmentation, which includes image scaling with different scales, flipping, adding salt and pepper noise and darkening. These implementations enlarged data size to 8 times than its original size,  which helps alot. Below are some sample images:

###### Scaling
![alt text][image6]

###### Flipping
![alt text][image7]

###### Salt and pepper noise
![alt text][image8]

###### Darkening
![alt text][image9]

### Start
##### Implement
Implement the code in the `train_seg.py` to train the model.
##### Run
Run the following command to run the project:
```
python train_seg.py
```
### Sample Output Image
![alt text][image1]

![alt text][image2]

![alt text][image3]

### Video Output

![alt text][image4]