# jimmyjohn
Using the [object_detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) of the [Tensorflow](https://www.tensorflow.org/) machine learning framework, we train a neural network model to detect a Jimmy John's delivery vehicle on a IP surveillance camera.  The model is then exported to run on a [Movidius Neural Compute](https://developer.movidius.com/) device to perform detections on a IP surveillance camera RTSP video stream in realtime. 

# Objectives
 - Learn the tensorflow object_detection API and training pipeline.
 - Determine if there is any _value_ in detecting custom objects in a fixed camera view.
 - What is the minimum accuracy needed to make object detection valuable?
 - How much training data is needed to produce desired accuracy?
 - What steps are necessary to augment a pre-trained detection model with custom data?
 - How does model selection affect accuracy and inference speed?
 - Is it even necessary to use a pre-trained model for custom objects? 
 
# Environment Setup
For collecting an image set and performing bounding-box annotations, any desktop
system can be used.  Since detection model training is compute-intensive and I did not have
access to a local CUDA/GPU machine, I chose to perform training on an Amazon EC2 [p3.2xlarge](https://aws.amazon.com/ec2/instance-types/p3/) instance.
The specific AMI I selected was the [NVIDIA Volta Deep Learning AMI](https://aws.amazon.com/marketplace/pp/B076K31M1S?qid=1511380876514&sr=0-1&ref_=srh_res_product_title) which is 
provided by [Nvidia GPU Compute Cloud](http://docs.nvidia.com/ngc/ngc-aws-setup-guide/index.html). This AMI is pre-configured with tensorflow at a cost of $3.305/hr.
  
For access to the tensorflow Nvidia NGC docker registry, I created a free account at [Nvidia NGC](https://ngc.nvidia.com/) so I could login
to their docker registry and pull the latest version into the AMI.  The NGC registry provides docker containers for many
popular deep learning frameworks.

# The Dataset
This is what a Jimmy Johns delivery looks like to the camera.
![Jimmy Johns Delivery](dataset/images/jimmy_johns/Devtown%20South%20Lot_20171027_131029_1.jpg?raw=true)

[Tensorflow recommends](https://www.tensorflow.org/tutorials/image_retraining) this for images:

For training to work well, you should gather at least a hundred photos of each kind of object you want to recognize. The more you can gather, the better the accuracy of your trained model is likely to be. You also need to make sure that the photos are a good representation of what your application will actually encounter. For example, if you take all your photos indoors against a blank wall and your users are trying to recognize objects outdoors, you probably won't see good results when you deploy.

Images were acquired from a surveillance video recorder, during sunny and cloudy and rainy days.  Annotations (labels, bounding boxes) were generated manually using the [labelImg](https://github.com/tzutalin/labelImg) utility, which creates a [PASCAL-VOC](http://host.robots.ox.ac.uk/pascal/VOC/) formatted XML file for each image.  There are actually 3 image classes that we are interested in:
- Jimmy Johns
- FedEx truck
- UPS truck

Tensorflow supports model training data to use [TFRecord](https://www.tensorflow.org/versions/r0.12/api_docs/python/python_io/#tfrecords_format_details) format.  So, we have to use the `create_tf_record.py` script to create these.
The script will iterate over the entire images dataset and partition 70% of the images as a training set `train.record`, and 30% as an evaluation set `eval.record`.  These files contain binary jpeg images along with their object class labels
 and bounding box annotations, in serialized [google protobuf](https://developers.google.com/protocol-buffers/) format.

# Training the model
The first model selected for object detection is [ssd_inception_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_08.tar.gz) from the tensorflow model zoo.
 
# References
These links were helpful to understand and guide me through the experiment.
 - Sentdex [Introduction and Use - Tensorflow Object Detection API Tutorial](https://pythonprogramming.net/introduction-use-tensorflow-object-detection-api-tutorial/)
 - Dat Tran [How to train your own Object Detector with TensorFlow’s Object Detector API](https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9)
 - O'Reilly [Object detection with TensorFlow](https://www.oreilly.com/ideas/object-detection-with-tensorflow)
