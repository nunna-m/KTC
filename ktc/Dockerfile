FROM tensorflow/tensorflow:2.4.0-gpu
RUN apt update
RUN apt install -y git python3-pip libsm6 libxext6 libxrender-dev libgl1-mesa-glx
RUN git clone https://github.com/nunna-m/KidneyTumorClassification
WORKDIR KidneyTumorClassification
RUN pip3 install -r requirements.txt
CMD NCCL_DEBUG=WARN NCCL_SHM_DISABLE=1 python3 -m ktc train\
    --config\
        configfiles/unet.yaml\
        configfiles/extra/data_options.yaml\
        configfiles/extra/deploy_options.yaml\
    --save_path\
        /kw_resources/results/kidney_tumor\
    --data_path\
        /kw_resources/datasets/kt_new_trainvaltest\
    --max_steps 10\
    --validate