FROM nvcr.io/nvidia/tritonserver:23.12-py3

COPY model-registry /model-registry

EXPOSE 8000
EXPOSE 8001
EXPOSE 8002

ENTRYPOINT [ "tritonserver", "--model-repository", "/model-registry" ]