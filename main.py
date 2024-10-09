import cv2
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from concurrent.futures import ThreadPoolExecutor

frames_without_person = list()


def setup_predictor():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    return DefaultPredictor(cfg)


def contains_person(outputs):
    classes = outputs["instances"].pred_classes
    return any(c == 0 for c in classes)


def process_frame(frame, frame_number, predictor):
    outputs = predictor(frame)
    if not contains_person(outputs):
        frames_without_person.append(frame_number)
        print(f"Сохранил кадр: {frame_number}")


def process_video(video_path, skip_frames=5):
    predictor = setup_predictor()
    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    executor = ThreadPoolExecutor(max_workers=4)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_number % skip_frames == 0:
            executor.submit(process_frame, frame, frame_number, predictor)
        frame_number += 1

    cap.release()
    executor.shutdown(wait=True)


def make_video_without_person(video_path, output_video_path):
    cap = cv2.VideoCapture(video_path)
    first_frame = cap.read()[1]
    height, width, layers = first_frame.shape
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(filename=output_video_path, fourcc=codec, fps=60, frameSize=(width, height))
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_number - frame_number % 60) in frames_without_person:
            video_writer.write(frame)
            print(f'Записался фрейм под номером {frame_number}')
        frame_number += 1

    cap.release()
    video_writer.release()


video_path = 'task1_2.mp4'
output_video_path = 'output_video.mp4'
process_video(video_path, skip_frames=60)
make_video_without_person(video_path, output_video_path)
