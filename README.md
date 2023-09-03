# ai_nvdia
# Clothes Detection

 The goal of this project is to create an image classification model that can accurately detect the type of clothes/attire presented to it(limited to 1 item) within an image

![add image descrition here](direct image link here)

## The Algorithm

The base model used for this project is resnet18 image detection, which is a model that scans and analyzes images in small segments. By doing so, the model is able to recognize patterns, and through multiple epochs of training the model is able to connect a certain pattern to a certain attribute, effectively increasing its ability to detect whatever is in the image. By specifying our data and detection targets down to clothes, the model is trained in a way that makes it a unique "clothing detection" model. 

## Running this project

1. Add steps for running this project.
2. Make sure to include any required libraries that need to be installed for your project to run.

### use this line to run the actual model
imagenet.py --model=resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=labels.txt test_photos/shoes.jpg output.jpg

# webcam ver
imagenet.py --model=resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=labels.txt /dev/video webrtc://@:8554/output
###

[View a video explanation here](video link)
