from locust import HttpUser, task, between
import json


payload = {}

headers = {
    'accept': 'application/json',
}

# Creating an API User class inheriting from Locust's HttpUser class
class APIUser(HttpUser):
    host = "http://localhost:8080"

    # Defining the post task using the JSON test data
    @task()
    def predict_endpoint(self):

        def _get_image_part(file_path, file_content_type='image/jpeg'):
            import os
            file_name = os.path.basename(file_path)
            file_content = open(file_path, 'rb')
            return file_name, file_content, file_content_type

        
        files = {
            "image_file" : _get_image_part("docs/yolo-test-people.jpeg")
        }

        ### stresstest graduation loan dev
        self.client.post('/v1/models/yolov8n', json=payload, headers=headers, files=files)