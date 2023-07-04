import cv2
import datetime
import imutils
import numpy as np
import time
from centroidtracker import CentroidTracker

confThreshold = 0.55
nmsThreshold = 0.2

#path1 ="C:\\Users\\manish.kumar\\Desktop\hackathon\\faceCount_AI Innovators\\data\\face-demographics-walking-and-pause.mp4"
#path1 ="C:\\Users\\manish.kumar\\Desktop\hackathon\\faceCount_AI Innovators\\data\\screen3.mp4"
#path1 ="C:\\Users\\manish.kumar\\Desktop\hackathon\\faceCount_AI Innovators\\data\\VN20230703_153937.mp4"
path1 ="C:\\Users\\manish.kumar\\Desktop\hackathon\\faceCount_AI Innovators\\data\\VID_20230703_175520.mp4"

# load the COCO class labels our YOLO model was trained on
labelsPath = 'C:\\Users\\manish.kumar\\Desktop\hackathon\\faceCount_AI Innovators\\model\\coco.names'
# derive the paths to the YOLO weights and model configuration
weightsPath = 'C:\\Users\\manish.kumar\\Desktop\hackathon\\faceCount_AI Innovators\\model\\face-yolov3-tiny_41000.weights'
configPath = 'C:\\Users\\manish.kumar\\Desktop\hackathon\\faceCount_AI Innovators\\model\\face-yolov3-tiny.cfg'

# centroid
tracker = CentroidTracker(maxDisappeared=100, maxDistance=90)

LABELS = []
with open(labelsPath, "r") as f:
    LABELS = f.read().strip("\n").split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
cap = cv2.VideoCapture(path1)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

def generate_boxes_confidences_classids(layerOutputs, H, W, confThreshold):
    boxes = []
    confidences = []
    classIDs = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confThreshold:
                if LABELS[classID] != "person":
                    continue
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    return boxes, confidences, classIDs

# FPS
fps_start_time = datetime.datetime.now()
fps = 0
total_frames = 0
writer = None

objectId_ls = []
# list to keep track of object IDs that were not detected in the current frame
not_detected_ids = list(objectId_ls)

fourcc_codec = cv2.VideoWriter_fourcc(*'XVID')
fps = 10.0
capture_size = (int(cap.get(3)), int(cap.get(4)))

out_ = cv2.VideoWriter("C:\\Users\\manish.kumar\\Desktop\\hackathon\\faceCount_AI Innovators\\output\\processed_VID_20230703_175520_.mp4", fourcc_codec, fps, capture_size)

face_count = {}

while True:
    ret, image = cap.read()
    # image = imutils.resize(image, width=600)
    total_frames = total_frames + 1

    if not ret:
        break

    (H, W) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)

    net.setInput(blob)
    start = time.time()

    layerOutputs = net.forward(ln)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))
    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    rects = []
    boxes, confidences, classIDs = generate_boxes_confidences_classids(layerOutputs, H, W, 0.5)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

    # ensure at least one detection exists

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            rects.append((x, y, w, h))
            # cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # list to keep track of object IDs that were not detected in the current frame
    not_detected_ids = list(objectId_ls)
    # tracker
    objects = tracker.update(rects)
    print(objects)
    # objectId_ls =[]
    for (objectId, bbox) in objects.items():
        x1, y1, x2, y2 = bbox
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        if objectId not in objectId_ls:
            objectId_ls.append(objectId)
        # remove the object ID from the list of IDs that were not detected
        if objectId in not_detected_ids:
            not_detected_ids.remove(objectId)

        cv2.rectangle(image, (x1, y1), (x1 + x2, y1 + y2), (0, 0, 255), 2)
        #text = "ID:{}".format(objectId)
        text ="person"
        cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

    
    # remove the IDs that were not detected from the total count
    objectId_ls = [obj_id for obj_id in objectId_ls if obj_id not in not_detected_ids]

    # count current person count and total person count
    opc_count = len(objectId_ls)
    opc_txt = "Count: {}".format(opc_count)
    cv2.putText(image, opc_txt, (5, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

    # FPS
    fps_end_time = datetime.datetime.now()
    time_diff = fps_end_time - fps_start_time
    if time_diff.seconds == 0:
        fps = 0.0
    else:
        fps = (total_frames / time_diff.seconds)

    fps_text = "FPS: {:.2f}".format(fps)
    #cv2.putText(image, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1)
    # show the output image
    out_.write(image)
    cv2.imshow("Image", image)
    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out_.release()
cv2.destroyAllWindows()
