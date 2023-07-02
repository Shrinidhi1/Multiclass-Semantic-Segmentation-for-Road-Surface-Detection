import gradio as gr
import numpy as np
import keras
from keras.models import load_model
from keras.metrics import MeanIoU
import cv2
from diffusers.utils import export_to_video

def custom_loss_func():
    focal_loss = sm.losses.CategoricalFocalLoss()
    return focal_loss

def iou_score():
    iou = sm.metrics.IOUScore(threshold=0.5)
    return iou

def fscore():
    fscore = sm.metrics.FScore(threshold=0.5)
    return fscore

model = load_model("resnetUnet.hdf5", custom_objects={'focal_loss': custom_loss_func, 'iou_score': iou_score, 'f1-score': fscore})

def get_prediction(img):
    img = img.reshape(1, 256, 256, 3)
    y_pred = model.predict(img)
    y_pred = np.argmax(y_pred, axis=3)
    y_pred = y_pred.reshape(256, 256)
    return y_pred

def add_color(img):
    img = img.reshape(256, 256, 1)
    frame = np.concatenate((img, img, img), axis=2)
    width = frame.shape[0]
    height = frame.shape[1]
    for x in range(width):
        for y in range(height):
            b, g, r = frame[x, y]
            if (b, g, r) == (0, 0, 0):  # background
                frame[x, y] = (0, 0, 0)
            elif (b, g, r) == (1, 1, 1):  # roadAsphalt
                frame[x, y] = (85, 85, 255)
            elif (b, g, r) == (2, 2, 2):  # roadPaved
                frame[x, y] = (85, 170, 127)
            elif (b, g, r) == (3, 3, 3):  # roadUnpaved
                frame[x, y] = (255, 170, 127)
            elif (b, g, r) == (4, 4, 4):  # roadMarking
                frame[x, y] = (255, 255, 255)
            elif (b, g, r) == (5, 5, 5):  # speedBump
                frame[x, y] = (255, 85, 255)
            elif (b, g, r) == (6, 6, 6):  # catsEye
                frame[x, y] = (255, 255, 127)
            elif (b, g, r) == (7, 7, 7):  # stormDrain
                frame[x, y] = (170, 0, 127)
            elif (b, g, r) == (8, 8, 8):  # manholeCover
                frame[x, y] = (0, 255, 255)
            elif (b, g, r) == (9, 9, 9):  # patchs
                frame[x, y] = (0, 0, 127)
            elif (b, g, r) == (10, 10, 10):  # waterPuddle
                frame[x, y] = (170, 0, 0)
            elif (b, g, r) == (11, 11, 11):  # pothole
                frame[x, y] = (255, 0, 0)
            elif (b, g, r) == (12, 12, 12):  # cracks
                frame[x, y] = (255, 85, 0)
    return frame


def main(Video):
    video = cv2.VideoCapture(Video)
    frames = []
    success = True
    while success:
        success, image = video.read()
        if success:
            image = cv2.resize(image, (256, 256))
            image = image.astype(np.uint8)
            pred = get_prediction(image)
            img = add_color(pred)
            img = img.astype('uint8')
            masked_img = cv2.addWeighted(image, 0.6, img, 0.8, 1)
            frames.append(masked_img)
    video.release()
    
    output_video = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25.0, (256, 256))
    for frame in frames:
        output_video.write(frame)
    output_video.release()
    
    output_video_path = 'output_video.mp4'
    return output_video_path

iface = gr.Interface(
    fn=main,
    inputs="video",
    outputs="video",
    capture_session=True,
    title="Road Surface Detection",
    description="Segment and visualize 12 road elements like road paved, road unpaved, road marking, speed bump, water puddles, potholes, patches, cracks, storm drain, road asphalt, manhole cover and cats eye in a video.",
    video_flag=True,
    css="""
        body {
            background-color: #f5f5f5;
            font-family: Arial, sans-serif;
        }
        .input-container {
            padding: 20px;
            margin-bottom: 20px;
            background-color: #ffffff;
            box-shadow: 0px 2px 10px rgba(0, 0, 0, 0.1);
        }
        .output-video-container {
            max-width: 100%;
            box-shadow: 0px 2px 10px rgba(0, 0, 0, 0.1);
        }
    """
)
iface.launch()