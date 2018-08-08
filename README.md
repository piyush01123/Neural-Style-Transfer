# neural-style
This repository contains the server-side code for my app that does Neural Style Transfer using deep CNNs, as mentioned in the paper [A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576.pdf) by Gatys et al., 2015. Android app repository of the project is [here](https://github.com/piyush-kgp/s3image).

Deep learning framework used: [Tensorflow](https://www.tensorflow.org)

CNN architecture used: [VGG19](https://arxiv.org/pdf/1409.1556.pdf), Simonyan and Zisserman, 2015. The weights can be obtained from [here](http://www.vlfeat.org/matconvnet/pretrained/).

#### How it works:
The entire repository has to be hosted on a server so that it continuously keeps listening for API requests being made. I hosted it using a free-tier AWS EC2 instance. <code>python neural_style_upload.py</code> when called from <code>crontab</code>/manually starts listening on the host URL where the code has been uploaded (which is <code>0.0.0.0:80</code> for the host but server's URL for everyone else). Upon passing URL of an image through an API call from a client, this code then saves the image as <code>image_from_url.jpg</code> on the server and calls functions from <code>neural_style_wrapper.py</code> and <code>nst_utls2.py</code> to generate an image and save it. Then, <code>neural_style_upload.py</code> uploads the generated image in an AWS S3 bucket (I have made my bucket and its contents public so that it can be accessed from code. To do this, AWS requires you to update the <code>license.lic</code>) and returns a JSON containing the URL of the image in the bucket to the client.

## Credits:
This implementation of the algorithm draws inspiration from a Programming assignment in the [CNN course](https://www.coursera.org/learn/convolutional-neural-networks/) of Coursera's [Deep Learning Specialization](https://www.coursera.org/specialization/deeplearning.ai).

## VGG-19
[Download](https://drive.google.com/open?id=1V2LAuCFv8qTBF3Vf792d9pDReHcx67TO)
