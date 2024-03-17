# Triton Inference Server Deployment

## Step to create AI/ML deployment with Triton Inverence Server :

### Download Model from Public Repo
```
yogiwahyuromadon@ADMINs-Air-3 raw-model % wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt
--2024-03-16 11:42:38--  https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt
Resolving github.com (github.com)... 20.205.243.166
Connecting to github.com (github.com)|20.205.243.166|:443... connected.
HTTP request sent, awaiting response... 302 Found
Location: https://objects.githubusercontent.com/github-production-release-asset-2e65be/521807533/6439a642-24f3-4fec-a962-1a97b22656d0?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240316%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240316T044238Z&X-Amz-Expires=300&X-Amz-Signature=ec0133bcbd07c838b3b7c143941c72815a038bc3f0be3fe032ffa967d211580b&X-Amz-SignedHeaders=host&actor_id=0&key_id=0&repo_id=521807533&response-content-disposition=attachment%3B%20filename%3Dyolov8n.pt&response-content-type=application%2Foctet-stream [following]
--2024-03-16 11:42:38--  https://objects.githubusercontent.com/github-production-release-asset-2e65be/521807533/6439a642-24f3-4fec-a962-1a97b22656d0?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240316%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240316T044238Z&X-Amz-Expires=300&X-Amz-Signature=ec0133bcbd07c838b3b7c143941c72815a038bc3f0be3fe032ffa967d211580b&X-Amz-SignedHeaders=host&actor_id=0&key_id=0&repo_id=521807533&response-content-disposition=attachment%3B%20filename%3Dyolov8n.pt&response-content-type=application%2Foctet-stream
Resolving objects.githubusercontent.com (objects.githubusercontent.com)... 185.199.109.133, 185.199.111.133, 185.199.108.133, ...
Connecting to objects.githubusercontent.com (objects.githubusercontent.com)|185.199.109.133|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 6534387 (6.2M) [application/octet-stream]
Saving to: ‘yolov8n.pt’

yolov8n.pt                                             100%[==========================================================================================================================>]   6.23M  1.70MB/s    in 3.6s    

2024-03-16 11:42:42 (1.71 MB/s) - ‘yolov8n.pt’ saved [6534387/6534387]

yogiwahyuromadon@ADMINs-Air-3 raw-model % 
```
### Convert From Pytorch to ONNX
Convert from `.pt` to `ONNX` model : 

 a. Install needed libraries :

    
    pip install --upgrade -r requirements.txt --no-cache-dir
    
b. Run notebook `converter.ipynb` to convert the model, <a href='https://docs.ultralytics.com/integrations/onnx/#usage'>reference.</a>
    
The output will ONNX model named `model.onnx`

### Triton Model Serving Preparation
To serve Model on Triton, you need to structurize the folder like this, <a href='https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html#repository-layout'>reference.</a> : 

```
  <model-repository-path>/
    <model-name>/
      [config.pbtxt]
      [<output-labels-file> ...]
      <version>/
        <model-definition-file>
      <version>/
        <model-definition-file>
      ...
    <model-name>/
      [config.pbtxt]
      [<output-labels-file> ...]
      <version>/
        <model-definition-file>
      <version>/
        <model-definition-file>
      ...
    ...
```

*Note :*
- If you don't know the exact model architecture, the file `config.pbtxt` will autogenerate by Triton.


### Build Triton Model Server Image

> In this step I use no GPU device because on my MacBook doesn't have dedicated Nvidia GPU, but I'll also provide how to run the Triton Model Server on GPU 

*Notes :*
- Before you run Nvidia GPU container, please make sure you already install the <a href="https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#driver-installation">Nvidia GPU Driver</a>, <a href="https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html">CUDA compatible</a>, and <a href='https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html'>NVIDIA Container Toolkit</a>

- Is also recommended to create Docker netwrok


1. On directory  `ai-test-case/triton-model-server`, build the Triton Servable Model Endpoint

    ```
    docker build -t triton-inference-server-model:latest .
    ```
2. After complete build the image, run the Triton Model Server :
    - on CPU device :
    ```
    docker run --name triton-model-server -p 8000:8000 -p 8001:8001 -p 8002:8002 triton-inference-server-model:latest
    ```
    - on GPU device(on all GPU) :
    ```
    docker run --name triton-model-server --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 triton-inference-server-model:latest
    ```
3. Output log 

    ```
    yogiwahyuromadon@ADMINs-Air-3 triton-model-server % docker run --name triton-model-server -p 8000:8000 -p 8001:8001 -p 8002:8002 triton-inference-server-model:latest           
    W0316 07:48:38.533692 1 pinned_memory_manager.cc:237] Unable to allocate pinned system memory, pinned memory pool will not be available: CUDA driver version is insufficient for CUDA runtime version
    I0316 07:48:38.533760 1 cuda_memory_manager.cc:117] CUDA memory pool disabled
    E0316 07:48:38.533850 1 server.cc:243] CudaDriverHelper has not been initialized.
    I0316 07:48:38.536204 1 model_lifecycle.cc:461] loading: yolov8n:1
    I0316 07:48:38.554627 1 onnxruntime.cc:2608] TRITONBACKEND_Initialize: onnxruntime
    I0316 07:48:38.554674 1 onnxruntime.cc:2618] Triton TRITONBACKEND API version: 1.17
    I0316 07:48:38.554677 1 onnxruntime.cc:2624] 'onnxruntime' TRITONBACKEND API version: 1.17
    I0316 07:48:38.554680 1 onnxruntime.cc:2654] backend configuration:
    {"cmdline":{"auto-complete-config":"true","backend-directory":"/opt/tritonserver/backends","min-compute-capability":"6.000000","default-max-batch-size":"4"}}
    I0316 07:48:38.573689 1 onnxruntime.cc:2719] TRITONBACKEND_ModelInitialize: yolov8n (version 1)
    I0316 07:48:38.647086 1 onnxruntime.cc:2784] TRITONBACKEND_ModelInstanceInitialize: yolov8n_0 (CPU device 0)
    I0316 07:48:38.647094 1 onnxruntime.cc:2784] TRITONBACKEND_ModelInstanceInitialize: yolov8n_1 (CPU device 0)
    I0316 07:48:38.724348 1 model_lifecycle.cc:818] successfully loaded 'yolov8n'
    I0316 07:48:38.725233 1 server.cc:606] 
    +------------------+------+
    | Repository Agent | Path |
    +------------------+------+
    +------------------+------+

    I0316 07:48:38.725279 1 server.cc:633] 
    +-------------+-----------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | Backend     | Path                                                            | Config                                                                                                                                                        |
    +-------------+-----------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | onnxruntime | /opt/tritonserver/backends/onnxruntime/libtriton_onnxruntime.so | {"cmdline":{"auto-complete-config":"true","backend-directory":"/opt/tritonserver/backends","min-compute-capability":"6.000000","default-max-batch-size":"4"}} |
    +-------------+-----------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------+

    I0316 07:48:38.725291 1 server.cc:676] 
    +---------+---------+--------+
    | Model   | Version | Status |
    +---------+---------+--------+
    | yolov8n | 1       | READY  |
    +---------+---------+--------+

    Error: Failed to initialize NVML
    W0316 07:48:38.738742 1 metrics.cc:738] DCGM unable to start: DCGM initialization error
    I0316 07:48:38.738964 1 metrics.cc:710] Collecting CPU metrics
    I0316 07:48:38.739070 1 tritonserver.cc:2483] 
    +----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | Option                           | Value                                                                                                                                                                                                           |
    +----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | server_id                        | triton                                                                                                                                                                                                          |
    | server_version                   | 2.41.0                                                                                                                                                                                                          |
    | server_extensions                | classification sequence model_repository model_repository(unload_dependents) schedule_policy model_configuration system_shared_memory cuda_shared_memory binary_tensor_data parameters statistics trace logging |
    | model_repository_path[0]         | /model-registry                                                                                                                                                                                                 |
    | model_control_mode               | MODE_NONE                                                                                                                                                                                                       |
    | strict_model_config              | 0                                                                                                                                                                                                               |
    | rate_limit                       | OFF                                                                                                                                                                                                             |
    | pinned_memory_pool_byte_size     | 268435456                                                                                                                                                                                                       |
    | min_supported_compute_capability | 6.0                                                                                                                                                                                                             |
    | strict_readiness                 | 1                                                                                                                                                                                                               |
    | exit_timeout                     | 30                                                                                                                                                                                                              |
    | cache_enabled                    | 0                                                                                                                                                                                                               |
    +----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

    I0316 07:48:38.754747 1 grpc_server.cc:2495] Started GRPCInferenceService at 0.0.0.0:8001
    I0316 07:48:38.754979 1 http_server.cc:4619] Started HTTPService at 0.0.0.0:8000
    I0316 07:48:38.804591 1 http_server.cc:282] Started Metrics Service at 0.0.0.0:8002
    ```

4. Health check Triton Model Server

<img src='./../docs/triton-yolo-health-check.png'>