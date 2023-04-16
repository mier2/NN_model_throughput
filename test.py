import time
import torch
from ray.air import session, Checkpoint, ScalingConfig, Result
from models import PolicyNN_23, PolicyNN_24, PolicyNN_48, PolicyNN_60, PolicyNN_108, PolicyNN_211
from ray.train.torch import TorchTrainer
import torch
from ray.air import session, Checkpoint, ScalingConfig, Result
from ray import train
import torch.distributed as dist
import csv


def train_func(config):
    #define the training model
    model = train.torch.prepare_model(config["modelName"])
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    #generate toy data
    input_size = 2048
    input_dim = config["dim_num"]
    dummy_input = torch.randn(input_size, input_dim, device='cuda', dtype=torch.float32)
    
    
    for i in range(config["num_epoch"]):
        _ = model(dummy_input)
        num_iters = 100
        start_time = time.time()
        for i in range(num_iters):
            optimizer.zero_grad()
            outputs = model(dummy_input)
            loss = criterion(outputs, torch.zeros(input_size, dtype=torch.long, device=f'cuda'))
            loss.backward()
            for param in model.parameters():
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            optimizer.step()
        
        end_time = time.time()
        throughput = num_iters * input_size / (end_time - start_time)
        checkpoint = Checkpoint.from_dict(
                dict(epoch=i)
                )
        session.report({'Throughput':throughput}, checkpoint=checkpoint)
        

def get_throughput(modelName, dimNum, labelDim):
    gpu_numbers = [1,2,4,8]
    for i in gpu_numbers:
        config = {"modelName": modelName, "num_epoch":1, "dim_num":dimNum, "label_dim":labelDim}
        trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=config,
        scaling_config=ScalingConfig(
            use_gpu=True,
            num_workers=i,
            )
        )
        result: Result = trainer.fit()
        data_23.append(result.metrics.get("Throughput"))

data_23 = []
data_24 = []
data_48 = []
data_60 = []
data_108 = []
data_211 = []
gpu_numbers=[1,2,4,8]    
#run the training model
get_throughput(PolicyNN_23(), 23, 9)
get_throughput(PolicyNN_24(), 24, 3)
get_throughput(PolicyNN_48(), 48, 12)
get_throughput(PolicyNN_60(), 60, 8)
get_throughput(PolicyNN_108(), 108, 21)
get_throughput(PolicyNN_211(), 211, 20)


    
    
    

#Write the csv file
#write the result to the csv file
with open("results.csv", "w", newline="") as csvfile:
    csv_writer= csv.writer(csvfile)
    csv_writer.writerow(['GPU Numbers', 'PolicyNN_23', 'PolicyNN_24', 'PolicyNN_48', 'PolicyNN_60', 'PolicyNN_108', 'PolicyNN_211'])
    for gpu_number, d_23, d_24, d_48, d_60, d_108, d_211 in zip(gpu_numbers, data_23, data_24, data_48, data_60, data_108, data_211):
        csv_writer.writerow([gpu_number, d_23, d_24, d_48, d_60, d_108, d_211])