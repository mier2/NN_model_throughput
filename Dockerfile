FROM rayproject/ray-ml
WORKDIR /NN_model_throughput
COPY . /NN_model_throughput


RUN pip install -r requirements.txt