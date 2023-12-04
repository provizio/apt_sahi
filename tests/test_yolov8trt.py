import unittest

import cv2
import tensorrt as trt

TRT_LOGGER = trt.Logger()

from sahi.utils.cv import read_image
from sahi.utils.yolov8trt import Yolov8TRTTestConstants, download_yolov8_trt_model

MODEL_DEVICE = "cuda:0"
CONFIDENCE_THRESHOLD = 0.4
IOU_THRESHOLD = 0.7
IMAGE_SIZE = 640


class TestYolov8TRTDetectionModel(unittest.TestCase):

    def test_load_model(self):
        from sahi.models.yolov8trt import Yolov8TrtDetectionModel

        download_yolov8_trt_model()

        yolov8_trt_detection_model = Yolov8TrtDetectionModel(
            model_path=Yolov8TRTTestConstants.YOLOV8N_TRT_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            iou_threshold=IOU_THRESHOLD,
            device=MODEL_DEVICE,
            category_mapping={"0": "something"},
            load_at_init=False,
            input_shape = [1, 3, 512, 416],
            output_shape = [1, 11, 4368]
        )

        self.assertNotEqual(yolov8_trt_detection_model.model, None)

    def test_set_model(self):
        #import tensorrt as trt

        from sahi.models.yolov8trt import Yolov8TrtDetectionModel

        download_yolov8_trt_model()

        yolov8_trt_detection_model = Yolov8TrtDetectionModel(
            model_path=Yolov8TRTTestConstants.YOLOV8N_TRT_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            iou_threshold=IOU_THRESHOLD,
            device=MODEL_DEVICE,
            category_mapping={"0": "something"},
            load_at_init=False,
            input_shape = [1, 3, 512, 416],
            output_shape = [1, 11, 4368]
        )

        self.assertNotEqual(yolov8_trt_detection_model.model, None)

    def test_perform_inference(self):
        from sahi.models.yolov8trt import Yolov8TrtDetectionModel

        # Init model
        download_yolov8_trt_model()

        yolov8_trt_detection_model = Yolov8TrtDetectionModel(
            model_path=Yolov8TRTTestConstants.YOLOV8N_TRT_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            iou_threshold=IOU_THRESHOLD,
            device=MODEL_DEVICE,
            category_mapping={"0": "something"},
            load_at_init=False,
            image_size=IMAGE_SIZE,
            input_shape=[1, 3, 512, 416],
            output_shape=[1, 11, 4368]
        )

        # Prepare image
        image_path = "tests/data/small-vehicles1.jpeg"
        image = cv2.imread(image_path)
    
        # Perform inference
        yolov8_trt_detection_model.perform_inference(image)
        original_predictions = yolov8_trt_detection_model.original_predictions

        # Ensure there are predictions
        assert original_predictions is not None

        

if __name__ == "__main__":
    unittest.main()
