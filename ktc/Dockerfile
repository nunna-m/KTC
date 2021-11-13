FROM tensorflow/tensorflow:2.4.0-gpu
RUN apt update
RUN apt install -y git python3-pip libsm6 libxext6 libxrender-dev libgl1-mesa-glx
RUN git clone https://github.com/nunna-m/KidneyTumorClassification
WORKDIR KidneyTumorClassification
RUN pip3 install -r requirements.txt
CMD NCCL_DEBUG=WARN NCCL_SHM_DISABLE=1 python3 -m ktc train\
    --whichos\
        linux\
    --config\
        configfiles/extra/data_options.yaml\
    --modalities\
        am\
        tm\
    --max_steps 10\
    --validate