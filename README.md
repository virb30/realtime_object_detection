# Realtime Object Detection

In this project we are using a YOLO network pre-trained with the COCO Dataset to
detect objects of a Video Streaming (rtmp or webcam) or a Video File.

For demonstration purposes this application was programmed to detect the following 
object classes:
* person
* remote
* tvmonitor
* keyboard
* mouse

## Usage

Clone this repo:
`git clone `

navigate to project directory: cd `object_detection`

install dependencies: `pip install -r requirements.txt`

Run the script: `python main.py -i [streaming]`

### Arguments

This script accepts only one argument `-i or --input` and its required.

You can pass:
* a number to use a connected webcam;
* a rtmp address to use rtmp server as video source or
* a file path to detect objects of a video file.


### Example

![Object detection example](object_detection_example.gif)