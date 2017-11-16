# jimmyjohn
An experiment in Machine Learning object recognition, to detect a Jimmy John's delivery vehicle on a IP surveillance camera.  Tensorflow is the ML framework being used.

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
 
