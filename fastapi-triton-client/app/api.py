from fastapi import FastAPI, File, UploadFile, status, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from app.functions import TritonYOLOv8
from app import config
from app.customize_logging.custom_logging import CustomizeLogger
from PIL import Image

import json
import os
import logging
import requests
import datetime
import numpy as np
import cv2


sem_ver = "dev.build.1"

triton_yolo = TritonYOLOv8(
    triton_endpoint=config.triton_enpoint, 
    port=config.port, 
    model_name=config.model_name, 
    model_version=config.model_version
)

### define config logging format json file
logger = logging.getLogger(__name__)
config_path="app/customize_logging/logging_config.json"


# Initialize FastAPI application
def create_app(sem_ver) -> FastAPI:
    app = FastAPI(
        title='YOLO V8n API', 
        description="YOLO V8 Model deployment . . .", 
        version=sem_ver
    )
    logger = CustomizeLogger.make_logger(config_path)
    app.logger = logger
    return app

app = create_app(sem_ver)

### For handling general/custom request reponse
class ResponseException(Exception):
    def __init__(self, msg: str, status_code: int):
        self.msg = msg
        self.status_code = status_code

### fastApi custom handling with uvicorn response For handling URL is not accessible [504]
@app.exception_handler(ResponseException)
async def response_exception_handler(request: Request, exc: ResponseException):
    response_body_url_err = {
        "timestamp": str(datetime.datetime.now()),
        "message": str(exc.msg)
    }

    return JSONResponse(
        status_code=int(exc.status_code),
        content=exc.msg,
    )


# CORS Middleware, need for deployment on K8s(expose service as L4 - LoadBalancer)
origins = [
    "http://localhost",
    "http://localhost:80",
    "http://0.0.0.0:80",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Custom exception handlers
@app.exception_handler(404)
async def not_found(request: Request, exc: HTTPException):
    return JSONResponse(content={"message": "Page not found"}, status_code=404)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: requests, exc: RequestValidationError):
    validation_request_response = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "message": "field required",
        "data": {"column": ["ktp"]}
    }
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=jsonable_encoder(validation_request_response))


# route handlers
@app.get("/health", tags=["Health-Check"])
async def read_root(request: Request) -> dict:
    health_msg = {
        "status": "UP",
        "checks": [
            {
                "name": "Enpoint /v1/models/yolov8n",
                "status": "UP"
            },
            {
                "name": "Triton Endpoint /v2/models/{}/versions/{}".format(config.model_name, config.model_version),
                "status": ""
            }
        ]
    }

    model_status = triton_yolo.health_check()

    if model_status == 200:
        health_msg['checks'][1]['status'] = "UP"
        request.app.logger.info("Yolo Health response : {}".format(model_status))
        
    else:
        health_msg['checks'][1]['status'] = "DOWN"
        status_code='500'
        request.app.logger.warning("Yolo Health response : {}".format(model_status))
       
        raise ResponseException(
            msg=health_msg, 
            status_code=status_code
        )
    
    return health_msg


@app.post(
    "/v1/models/yolov8n",
    tags=["Predictions"],
    description="Get Prediction from Yolo V8n Detection Model. <br>From Image File"
)
async def yolo_inference(
    request: Request,
    image_file: UploadFile = File(...)
):
    image_binary = await image_file.read()
    file_name = image_file.filename
    result = triton_yolo.inference(image_binary=image_binary, file_name=file_name)

    return result
