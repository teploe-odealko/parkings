import torchvision.transforms as T
import cv2
import numpy as np
import torchvision
from PIL import Image
from app import engine
import ast
from sqlalchemy.sql import text


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.
    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou

def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps

class nnAPI:
    def __init__(self):
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.car_boxes = []
        self.COCO_INSTANCE_CATEGORY_NAMES = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    def model_output(self, img_cv2, threshold):
        img = Image.fromarray(img_cv2) # Load the image
        transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
        img = transform(img) # Apply the transform to the image
        pred = self.model([img]) # Pass the image to the model
        pred_class = [self.COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())] # Get the Prediction Score
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
        pred_score = list(pred[0]['scores'].detach().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] # Get list of index with score greater than threshold.
        pred_boxes = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]
        return pred_boxes, pred_class

    def get_car_boxes(self, boxes, class_ids):
        for i, box in enumerate(boxes):
    #         # Если найденный объект не автомобиль, то пропускаем его.
            if class_ids[i] in ['car', 'truck', 'bus']:
                self.car_boxes.append(box)
        tmp = []

        for box in boxes:
            tmp.append([item for t in box for item in t])
        self.car_boxes = tmp
        return self.car_boxes



    def make_prediction(self, img_cv2, camera_id, threshold=0.5, rect_th=1, text_size=3, text_th=3):
        # print(img_cv2)
        boxes, pred_cls = self.model_output(img_cv2, threshold) # Get predictions
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB) # Convert to RGB
        car_boxes = self.get_car_boxes(boxes, pred_cls)

        # Получаем из бд граунд трув парковки
        print("query camera id", camera_id)
        statement = text("""SELECT * FROM parking_boxes where camera_id={}""".format(camera_id))
        res = engine.connect().execute(statement)
        # print(list(res))
        # print(list(res)[0][1])
        parking_boxes_str = list(res)[0][1]
        # [[], []]
        parking_boxes = ast.literal_eval(parking_boxes_str)
        # print(car_boxes)
        overlaps = compute_overlaps(np.array(parking_boxes), np.array(car_boxes))

        free_space = False
        # print(np.array(car_boxes))

        # [[0.9], [0.1]
        #  [], []]
        for parking_area, overlap_areas in zip(np.array(parking_boxes), overlaps):

            # Ищем максимальное значение пересечения с любой обнаруженной
            # на кадре машиной (неважно, какой).
            max_IoU_overlap = np.max(overlap_areas)

            # Получаем верхнюю левую и нижнюю правую координаты парковочного места.
            x1, y1, x2, y2 = parking_area
            # Проверяем, свободно ли место, проверив значение IoU.
            if max_IoU_overlap < 0.15:
                # Место свободно! Рисуем зелёную рамку вокруг него.
                cv2.rectangle(img_cv2, tuple([int(x1), int(y1)]), tuple([int(x2), int(y2)]), (0, 255, 0), 2)
                # Отмечаем, что мы нашли как минимум оно свободное место.
                free_space = True
            else:
                # Место всё ещё занято — рисуем красную рамку.
                cv2.rectangle(img_cv2, tuple([int(x1), int(y1)]), tuple([int(x2), int(y2)]), (255, 0, 0), 2)

            # Записываем значение IoU внутри рамки.
            # font = cv2.FONT_HERSHEY_DUPLEX
            # cv2.putText(img_cv2, f"{max_IoU_overlap:0.2}", (x1 + 6, y2 - 6), font, 0.3, (255, 255, 255))



        # for i in range(len(car_boxes)):
        #     cv2.rectangle(img_cv2, tuple(car_boxes[i][:2]), tuple(car_boxes[i][2:]),color=(0, 255, 0), thickness=rect_th)

        return img_cv2
