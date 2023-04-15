# NN_model throughput


## Setup docker
```bash
#bulid the image
docker build -t <image_name> .

#run container based on this image
docker run --gpus all -it --name <container name> --user=root --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=10.24gb <image name>
```

## Run the models

```bash
python test_<model dim number>.py
```

## Get results
```bash
sudo docker cp <container name>:/NN_model_throughput/results.csv /path/to/your/directory
```