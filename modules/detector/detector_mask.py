import os
os.environ["OMP_NUM_THREADS"] = "1"
import torch

import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import matplotlib.pyplot as plt
import numpy as np

class Batchredictor(DefaultPredictor):
    def __call__(self, images):
        inputs = [
            {'image': torch.from_numpy(image.astype("float32").transpose(2, 0, 1))}
            for image in images
        ]
        with torch.no_grad():
            return self.model(inputs)

det_COI = [
        56,  # chair
        57,  # couch
        58,  # potted plant
        59,  # bed
        61,  # toilet
        62,  # tv
        60,  # dining table
        63,  # laptop
        68,  # microwave
        69,  # oven
        71,  # sink
        72,  # refrigerator
        74,  # clock
        75,  # vase
    ]


class Detector:
    def __init__(self, args, COI):

        # Create configs
        self.args = args
        self.device = f'cuda:{self.args.model_gpu}' if torch.cuda.is_available() else 'cpu'
        self.cfg = get_cfg()
        self.cfg.merge_from_file(args.detection_model + "/mask_rcnn_R_50_FPN_3x.yaml")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9  # set threshold for this model
        # self.cfg.MODEL.WEIGHTS = "model_final_f10217.pkl"
        self.cfg.MODEL.WEIGHTS = args.detection_model + "/model_mask50_FPN.pkl"
        self.cfg.INPUT.FORMAT = "RGB"
        self.cfg.DATALOADER.NUM_WORKERS = 1
        self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])

        self.COI = COI
        # Create predictor

        self.predictor = DefaultPredictor(self.cfg)
        self.predictor.model = self.predictor.model.to(self.device)
        # self.batch_predictor = Batchredictor(self.cfg)


    def predicted_img(self, input, batch=False, show=False):
        if batch:
            return self.batch_predicted_det(input)

        # Make prediction
        # input = input.to(self.device)
        torch.set_num_threads(1)
        outputs = self.predictor(input)

        pred_classes = outputs['instances']._fields['pred_classes'].cpu().numpy()
        valid_classes = np.isin(pred_classes, self.COI)
        # valid_classes = np.array([False, True])
        pred_classes = pred_classes[valid_classes]
        pred_classes = np.array([self.COI.index(i) for i in pred_classes])
        outputs['instances']._fields['pred_classes'] = outputs['instances']._fields['pred_classes'][valid_classes]


        outputs['instances']._fields['pred_masks'] = outputs['instances']._fields['pred_masks'][valid_classes]
        masks = outputs['instances']._fields['pred_masks'].cpu().numpy()

        outputs['instances']._fields['pred_boxes'] = outputs['instances']._fields['pred_boxes'][valid_classes]
        boxes = outputs['instances']._fields['pred_boxes'].tensor.cpu().numpy()

        outputs['instances']._fields['scores'] = outputs['instances']._fields['scores'][valid_classes]
        scores = outputs['instances']._fields['scores'].cpu().numpy()

        pred_out = np.zeros(len(self.COI))
        for i, cls in enumerate(pred_classes):
            if pred_out[cls] < scores[i]:
                pred_out[cls] = scores[i]

        if show:
            v = Visualizer(input[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            v = v.get_image()
            return cv2.cvtColor(v, cv2.COLOR_BGR2RGB), pred_classes, scores, pred_out, masks, boxes
        else:
            return pred_classes, scores, pred_out, masks, boxes

    def batch_predicted_det(self, images):
        # Make prediction
        outputs = self.batch_predictor(images)

        pred_classes_list = []
        scores_list = []
        pred_out_list = []
        masks_list = []
        boxes_list = []

        for output in outputs:

            pred_classes = output['instances']._fields['pred_classes'].cpu().numpy()
            valid_classes = np.isin(pred_classes, COI)
            # valid_classes = np.array([False, True])
            pred_classes = pred_classes[valid_classes]
            pred_classes = np.array([COI.index(i) for i in pred_classes])
            output['instances']._fields['pred_classes'] = output['instances']._fields['pred_classes'][valid_classes]

            output['instances']._fields['pred_masks'] = output['instances']._fields['pred_masks'][valid_classes]
            masks = output['instances']._fields['pred_masks'].cpu().numpy()

            output['instances']._fields['pred_boxes'] = output['instances']._fields['pred_boxes'][valid_classes]
            boxes = output['instances']._fields['pred_boxes'].tensor.cpu().numpy()

            output['instances']._fields['scores'] = output['instances']._fields['scores'][valid_classes]
            scores = output['instances']._fields['scores'].cpu().numpy()

            pred_out = np.zeros(len(COI))
            for i, cls in enumerate(pred_classes):
                if pred_out[cls] < scores[i]:
                    pred_out[cls] = scores[i]

            pred_classes_list.append(pred_classes)
            scores_list.append(scores)
            pred_out_list.append(pred_out)
            masks_list.append(masks)
            boxes_list.append(boxes)

        return pred_classes_list, scores_list, pred_out_list, masks_list, boxes_list



if __name__ == "__main__":
    import argparse
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "9"

    COI = [
        56,  # chair
        57,  # couch
        58,  # potted plant
        59,  # bed
        60,  # dining table
        61,  # toilet
        62,  # tv
        63,  # laptop
        68,  # microwave
        69,  # oven
        71,  # sink
        72,  # refrigerator
        74,  # clock
        75,  # vase
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_gpu", type=str, default="0")
    args = parser.parse_args()

    detector = Detector(args, COI)

    im = cv2.imread("example2.png")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # input_rgb = np.zeros_like(im)
    # input_rgb[:, :, 0] = im[:, :, 2]
    # input_rgb[:, :, 1] = im[:, :, 1]
    # input_rgb[:, :, 2] = im[:, :, 0]
    image, pred_classes, scores, pred_out, masks, boxex = detector.predicted_img(im, show=True)

    plt.imsave('test_det.png',image)
    # plt.show()