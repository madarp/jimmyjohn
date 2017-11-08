# jimmyjohn
An experiment in Machine Learning object recognition, to detect a Jimmy John's delivery vehicle on a IP surveillance camera.  Tensorflow is the ML framework being used.

# The Dataset
This is what a Jimmy Johns delivery looks like to the camera.
![Jimmy Johns Delivery](dataset/images/jimmy_johns/Devtown%20South%20Lot_20171027_131029_1.jpg?raw=true)

Images were acquired from a surveillance video recorder.  Annotations (labels, bounding boxes) were generated manually using the [labelImg](https://github.com/tzutalin/labelImg) utility, which creates a [PASCAL-VOC](http://host.robots.ox.ac.uk/pascal/VOC/) formatted XML file for each image.  There are actually 3 image classes that we are interested in:
- Jimmy Johns
- FedEx truck
- UPS truck
