import torch
from ray.air import session, Checkpoint, ScalingConfig, Result
from ray import train
from models import PolicyNN_60, PolicyNN_48, PolicyNN_24, PolicyNN_23, PolicyNN_108, PolicyNN_211
from ray.train.torch import TorchTrainer
import csv

#traning model function
def train_func(config):
    #define the training model
    model = train.torch.prepare_model(config["modelName"])
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    
    #generate toy data
    input = torch.randn(1000, config["dim_num"])
    labels = torch.randn(1000, config["label_dim"])
    dataset = torch.utils.data.TensorDataset(input, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"])
    dataloader = train.torch.prepare_data_loader(dataloader)
    
    
    for i in range(config["num_epoch"]):
        for X, y in dataloader:
            pred = model.forward(X)
            #compute loss
            loss = criterion(pred,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            state_dict = model.state_dict()
            checkpoint = Checkpoint.from_dict(
                dict(epoch=i,
                    model_weights=state_dict)
                )
            session.report({}, checkpoint=checkpoint)
        
        
        
    
    
#run the training model
config = {"modelName": PolicyNN_60(), "num_epoch":1, "dim_num":60, "label_dim":8, "batch_size": 1024}
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(
        use_gpu=True,
        num_workers=1,
        )
    )
result: Result = trainer.fit()

print(result.metrics)









batch_size = 1024
#ouput the throughput into a csv file
iterations = result.metrics.get('training_iteration')
time = result.metrics.get("time_total_s")
model_name = "PolicyNN_60"
num_gpu = 1

#calculate the throughput
total_samples = batch_size * iterations
throughput = total_samples/time

data = [
    ["Model Dim#", "Batch Size","Number of GPUS", "Throughput"],
    [model_name, num_gpu, throughput]
]

with open("results.csv", "w", newline="") as csvfile:
    csv_writer= csv.writer(csvfile)
    csv_writer.writerows(data)

    
    