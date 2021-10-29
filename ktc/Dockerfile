FROM tensorflow/tensorflow:2.4.0-gpu
RUN apt update
RUN apt install -y git python3-pip libsm6 libxext6 libxrender-dev libgl1-mesa-glx
RUN git clone https://github.com/nunna-m/KidneyTumorClassification
WORKDIR KidneyTumorClassification
RUN pip3 install -r requirements.txt
CMD NCCL_DEBUG=WARN NCCL_SHM_DISABLE=1 python3 -m annotator train\
    --config configs/unet.yaml\
    --save_path /kw_resources/results/annotation/db3/temp\
    --data_path\
        /kw_resources/datasets/projects/annotation/db3/train/cancer.tfrecords\
        /kw_resources/datasets/projects/annotation/db3/train/healthy.tfrecords\
    --max_steps 200000\
    --save_freq 5000\
    --val_data_path\
        /kw_resources/datasets/projects/annotation/db3/val/cancer.tfrecords\
        /kw_resources/datasets/projects/annotation/db3/val/healthy.tfrecords\
    --validate\
    --visualize