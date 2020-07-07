import re
import os
import cv2
import dlib
import time
import numpy as np

from imutils import video
from src.utils import LandmarkExtractor

def csv_reader(csv_path):
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]
        file_result = []
        for line in lines:
            values = line.split(',')
            # frame_number, id, lt, br, score
            left = int(re.findall(r'\d+', values[2])[0])
            top = int(re.findall(r'\d+', values[3])[0])
            right = int(re.findall(r'\d+', values[4])[0])
            bottom = int(re.findall(r'\d+', values[5])[0])
            face_box = dlib.rectangle(left, top, right, bottom)
            if values[1] == "Unknown":
                file_result.append({"frame":int(values[0]), "ID":values[1], "face_box":face_box})
            else:
                file_result.append({"frame":int(values[0]), "ID":int(values[1]), "face_box":face_box})
    return file_result

def detect_from_id(video_path, csv_file_result, dim = "2D"):
    result_npy_path = "../output/landmarks/{}/".format(video_path.split("/")[-1][:-4])
    result_img_path = "../output/landmarks/frames/{}/".format(video_path.split("/")[-1][:-4])
    if not os.path.exists(result_npy_path):
        os.makedirs(result_npy_path)
        os.makedirs(result_img_path)
    print(result_npy_path)
    cap = cv2.VideoCapture(video_path)
    fps = video.FPS().start()
    count = 0

    LE = LandmarkExtractor(dim=dim)

    if cap.isOpened() == False:
        print("Error opening video stream")

    while cap.isOpened():
        t = time.time()

        ret, frame = cap.read()
        for result in csv_file_result:
            if result["frame"] == count:
                print(result["face_box"])
                landmarks = LE.landmark_extractor(frame, [result["face_box"]])
                if not os.path.exists(os.path.join(result_npy_path, str(count))):
                    os.mkdir(os.path.join(result_npy_path, str(count)))
                np.save(os.path.join(result_npy_path, str(count), str(result["ID"])), landmarks)
                print("npy saved at ", result_npy_path, str(count), str(result["ID"]))
                cv2.imwrite(os.path.join(result_img_path, "{}_{}.png".format(count, result["ID"])),
                            LE.draw_landmark(frame, [result["face_box"]], landmarks))
                print("Saved {} video {} frame".format(video_path.split('/')[-1], count))

        count += 1
        fps.update()
        print('[INFO] {} frame elapsed time: {:.2f}'.format(count, time.time() - t))

        if count == 5000:
            break
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break
    fps.stop()
    cap.release()
    cv2.destroyAllWindows()

def landmark_reader(npy_path):
    """
    npy_path: outputs/landmarks/<video_name>
    output: dictionary npy_info
            for using numpy array npy_info[index]["landmarks"]
    """
    npy_info = []
    video_name = npy_path.split('/')[-1]
    for frames in os.listdir(npy_path):
        for ids in os.listdir(os.path.join(npy_path, frames)):
            if ids.endswith("npy"):
                npy_info.append({"VideoName":video_name, "frame_number":int(frames),
                                 "ID":int(ids[:-4]), "landmarks":np.load(os.path.join(npy_path, frames, ids))})
    return npy_info


example = landmark_reader("../output/landmarks/recording")
for info in example:
    print("video name: {}\nframe number: {}\nface id: {}, landmark shape: {}\n\n".format(
        info["VideoName"], info["frame_number"], info["ID"], info["landmarks"].shape))

