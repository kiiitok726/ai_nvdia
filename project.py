import sys
import argparse
from jetson_inference import imageNet
from jetson_utils import videoSource, videoOutput, cudaFont, Log

net = imageNet(model="resnet18.onnx", labels="labels.txt", 
input_blob="input_0", output_blob="output_0")

input1 = videoSource("/dev/video0", argv=sys.argv)
output = videoOutput("webrtc://@:8554/output", argv=sys.argv)
font = cudaFont()

while True:
    # capture the next image
    img = input1.Capture()

    if img is None: # timeout
        continue
    
    class_id, confidence = net.Classify(img)
    confidence *= 100

    classLabel = net.GetClassLabel(class_id)

    print(f"Classified as: {classLabel}, confidence = {confidence}")

    font.OverlayText(img, text=f"{confidence:05.2f}% {classLabel}", 
    x=5, y=5 * (font.GetSize() + 5), color=font.White, background=font.Gray40)

    output.Render(img)