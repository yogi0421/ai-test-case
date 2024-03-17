import requests
import cv2
import numpy as np
from ultralytics import YOLO
import json


class TritonYOLOv8:
    def __init__(self, triton_endpoint, port, model_name, model_version):
        self.triton_endpoint = triton_endpoint
        self.port = port
        self.model_name = model_name
        self.model_version = model_version
        self.model = YOLO('http://{}:{}/{}'.format(self.triton_endpoint, self.port, self.model_name), task='detect')

    def inference(self, image_binary, file_name):
        image_array = np.frombuffer(image_binary, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        results = self.model(img)
        infer_result = json.loads(results[0].tojson())

        split_name = file_name.split(".")
        str_name = "/tmp/inference-result/yolov8n_inference-{}.jpg".format(split_name[0])
        output_dir = results[0].save(filename=str_name)

        json_response = {
            "input_dir": file_name,
            "output_dir": output_dir,
            "data": infer_result
        }

        return json_response
    
    def health_check(self):
        try:
            response = requests.get(
                "http://{}:{}/v2/models/{}/versions/{}/ready".format(
                    self.triton_endpoint, 
                    self.port, 
                    self.model_name, 
                    self.model_version
                )
            )
        except Exception as e:
            print("Execption : ", e)
            return 500

        return response.status_code
