import torch
from ray.air import session, Checkpoint, ScalingConfig, Result
from ray import train
from models import PolicyNN_108
from ray.train.torch import TorchTrainer
import csv

data = [
    ["Model Dim#", "Batch Size","Number of GPUS", "Throughput"],
]
def group_res( model_name, batch_size, num_gpu, result):
    iterations = result.metrics.get('training_iteration')
    time = result.metrics.get("time_total_s")
    total_samples = batch_size * iterations
    throughput = total_samples/time
    return [model_name,batch_size, num_gpu, throughput]
    
    
    
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
#1GPU
#PolicyNN_108 with different batch size
config = {"modelName": PolicyNN_108(), "num_epoch":1, "dim_num":108, "label_dim":21, "batch_size": 1024}
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(
        use_gpu=True,
        num_workers=1,
        )
    )
result: Result = trainer.fit()


data.append(group_res("PolicyNN_108", 1024, 1, result))

config = {"modelName": PolicyNN_108(), "num_epoch":1, "dim_num":108, "label_dim":21, "batch_size": 2048}
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(
        use_gpu=True,
        num_workers=1,
        )
    )
result: Result = trainer.fit()

data.append(group_res("PolicyNN_108", 2048, 1, result))

config = {"modelName": PolicyNN_108(), "num_epoch":1, "dim_num":108, "label_dim":21, "batch_size": 4196}
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(
        use_gpu=True,
        num_workers=1,
        )
    )
result: Result = trainer.fit()

data.append(group_res("PolicyNN_108", 4196, 1, result))

config = {"modelName": PolicyNN_108(), "num_epoch":1, "dim_num":108, "label_dim":21, "batch_size": 8192}
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(
        use_gpu=True,
        num_workers=1,
        )
    )
result: Result = trainer.fit()

data.append(group_res("PolicyNN_108", 8192, 1, result))

config = {"modelName": PolicyNN_108(), "num_epoch":1, "dim_num":108, "label_dim":21, "batch_size": 16384}
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(
        use_gpu=True,
        num_workers=1,
        )
    )
result: Result = trainer.fit()

data.append(group_res("PolicyNN_108", 16384, 1, result))

#2GPU
#PolicyNN_108 with different batch size
config = {"modelName": PolicyNN_108(), "num_epoch":1, "dim_num":108, "label_dim":21, "batch_size": 1024}
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(
        use_gpu=True,
        num_workers=2,
        )
    )
result: Result = trainer.fit()

data.append(group_res("PolicyNN_108", 1024, 2, result))

config = {"modelName": PolicyNN_108(), "num_epoch":1, "dim_num":108, "label_dim":21, "batch_size": 2048}
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(
        use_gpu=True,
        num_workers=2,
        )
    )
result: Result = trainer.fit()

data.append(group_res("PolicyNN_108", 2048, 2, result))

config = {"modelName": PolicyNN_108(), "num_epoch":1, "dim_num":108, "label_dim":21, "batch_size": 4196}
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(
        use_gpu=True,
        num_workers=2,
        )
    )
result: Result = trainer.fit()

data.append(group_res("PolicyNN_108", 4196, 2, result))

config = {"modelName": PolicyNN_108(), "num_epoch":1, "dim_num":108, "label_dim":21, "batch_size": 8192}
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(
        use_gpu=True,
        num_workers=2,
        )
    )
result: Result = trainer.fit()

data.append(group_res("PolicyNN_108", 8192, 2, result))

config = {"modelName": PolicyNN_108(), "num_epoch":1, "dim_num":108, "label_dim":21, "batch_size": 16384}
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(
        use_gpu=True,
        num_workers=2,
        )
    )
result: Result = trainer.fit()

data.append(group_res("PolicyNN_108", 16384, 2, result))

#4GPU
#PolicyNN_108 with different batch size
config = {"modelName": PolicyNN_108(), "num_epoch":1, "dim_num":108, "label_dim":21, "batch_size": 1024}
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(
        use_gpu=True,
        num_workers=4,
        )
    )
result: Result = trainer.fit()


data.append(group_res("PolicyNN_108", 1024, 4, result))

config = {"modelName": PolicyNN_108(), "num_epoch":1, "dim_num":108, "label_dim":21, "batch_size": 2048}
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(
        use_gpu=True,
        num_workers=4,
        )
    )
result: Result = trainer.fit()

data.append(group_res("PolicyNN_108", 2048, 4, result))

config = {"modelName": PolicyNN_108(), "num_epoch":1, "dim_num":108, "label_dim":21, "batch_size": 4196}
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(
        use_gpu=True,
        num_workers=4,
        )
    )
result: Result = trainer.fit()

data.append(group_res("PolicyNN_108", 4196, 4, result))

config = {"modelName": PolicyNN_108(), "num_epoch":1, "dim_num":108, "label_dim":21, "batch_size": 8192}
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(
        use_gpu=True,
        num_workers=4,
        )
    )
result: Result = trainer.fit()

data.append(group_res("PolicyNN_108", 8192, 4, result))

config = {"modelName": PolicyNN_108(), "num_epoch":1, "dim_num":108, "label_dim":21, "batch_size": 16384}
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(
        use_gpu=True,
        num_workers=4,
        )
    )
result: Result = trainer.fit()

data.append(group_res("PolicyNN_108", 16384, 4, result))

#8GPU
#PolicyNN_108 with different batch size
config = {"modelName": PolicyNN_108(), "num_epoch":1, "dim_num":108, "label_dim":21, "batch_size": 1024}
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(
        use_gpu=True,
        num_workers=8,
        )
    )
result: Result = trainer.fit()


data.append(group_res("PolicyNN_108", 1024, 8, result))

config = {"modelName": PolicyNN_108(), "num_epoch":1, "dim_num":108, "label_dim":21, "batch_size": 2048}
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(
        use_gpu=True,
        num_workers=8,
        )
    )
result: Result = trainer.fit()

data.append(group_res("PolicyNN_108", 2048, 8, result))

config = {"modelName": PolicyNN_108(), "num_epoch":1, "dim_num":108, "label_dim":21, "batch_size": 4196}
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(
        use_gpu=True,
        num_workers=8,
        )
    )
result: Result = trainer.fit()

data.append(group_res("PolicyNN_108", 4196, 8, result))

config = {"modelName": PolicyNN_108(), "num_epoch":1, "dim_num":108, "label_dim":21, "batch_size": 8192}
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(
        use_gpu=True,
        num_workers=8,
        )
    )
result: Result = trainer.fit()

data.append(group_res("PolicyNN_108", 8192, 8, result))

config = {"modelName": PolicyNN_108(), "num_epoch":1, "dim_num":108, "label_dim":21, "batch_size": 16384}
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(
        use_gpu=True,
        num_workers=8,
        )
    )
result: Result = trainer.fit()

data.append(group_res("PolicyNN_108", 16384, 8, result))





#write the result to the csv file
with open("results.csv", "a", newline="") as csvfile:
    csv_writer= csv.writer(csvfile)
    csv_writer.writerows(data)