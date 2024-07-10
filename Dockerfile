FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel
ENV TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.2 7.5 8.0 8.6 8.7 9.0+PTX"
ENV PYTHONUNBUFFERED=TRUE

RUN apt-get update && apt-get install --no-install-recommends -y \
    libegl1 \
    libgl1 \
    libgomp1

COPY requirements.txt .
COPY external_libs ./external_libs
RUN pip install -r requirements.txt
RUN pip install jupyterlab

ENTRYPOINT [ "jupyter", "lab", "--ip=0.0.0.0", "--allow-root" ]