import torch
from ray.air import session, Checkpoint, ScalingConfig, Result
from models import PolicyNN_23, train_func, group_res
from ray.train.torch import TorchTrainer
import csv

data = []

           
#run the training model
#1GPU
#PolicyNN_23 with different batch size
config = {"modelName": PolicyNN_23(), "num_epoch":1, "dim_num":23, "label_dim":9, "batch_size": 1024}
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(
        use_gpu=True,
        num_workers=1,
        )
    )
result: Result = trainer.fit()


data.append(group_res("PolicyNN_23", 1024, 1, result))

config = {"modelName": PolicyNN_23(), "num_epoch":1, "dim_num":23, "label_dim":9, "batch_size": 2048}
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(
        use_gpu=True,
        num_workers=1,
        )
    )
result: Result = trainer.fit()

data.append(group_res("PolicyNN_23", 2048, 1, result))

config = {"modelName": PolicyNN_23(), "num_epoch":1, "dim_num":23, "label_dim":9, "batch_size": 4196}
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(
        use_gpu=True,
        num_workers=1,
        )
    )
result: Result = trainer.fit()

data.append(group_res("PolicyNN_23", 4196, 1, result))

config = {"modelName": PolicyNN_23(), "num_epoch":1, "dim_num":23, "label_dim":9, "batch_size": 8192}
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(
        use_gpu=True,
        num_workers=1,
        )
    )
result: Result = trainer.fit()

data.append(group_res("PolicyNN_23", 8192, 1, result))

config = {"modelName": PolicyNN_23(), "num_epoch":1, "dim_num":23, "label_dim":9, "batch_size": 16384}
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(
        use_gpu=True,
        num_workers=1,
        )
    )
result: Result = trainer.fit()

data.append(group_res("PolicyNN_23", 16384, 1, result))

#2GPU
#PolicyNN_23 with different batch size
config = {"modelName": PolicyNN_23(), "num_epoch":1, "dim_num":23, "label_dim":9, "batch_size": 1024}
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(
        use_gpu=True,
        num_workers=2,
        )
    )
result: Result = trainer.fit()

data.append(group_res("PolicyNN_23", 1024, 2, result))

config = {"modelName": PolicyNN_23(), "num_epoch":1, "dim_num":23, "label_dim":9, "batch_size": 2048}
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(
        use_gpu=True,
        num_workers=2,
        )
    )
result: Result = trainer.fit()

data.append(group_res("PolicyNN_23", 2048, 2, result))

config = {"modelName": PolicyNN_23(), "num_epoch":1, "dim_num":23, "label_dim":9, "batch_size": 4196}
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(
        use_gpu=True,
        num_workers=2,
        )
    )
result: Result = trainer.fit()

data.append(group_res("PolicyNN_23", 4196, 2, result))

config = {"modelName": PolicyNN_23(), "num_epoch":1, "dim_num":23, "label_dim":9, "batch_size": 8192}
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(
        use_gpu=True,
        num_workers=2,
        )
    )
result: Result = trainer.fit()

data.append(group_res("PolicyNN_23", 8192, 2, result))

config = {"modelName": PolicyNN_23(), "num_epoch":1, "dim_num":23, "label_dim":9, "batch_size": 16384}
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(
        use_gpu=True,
        num_workers=2,
        )
    )
result: Result = trainer.fit()

data.append(group_res("PolicyNN_23", 16384, 2, result))

#4GPU
#PolicyNN_23 with different batch size
config = {"modelName": PolicyNN_23(), "num_epoch":1, "dim_num":23, "label_dim":9, "batch_size": 1024}
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(
        use_gpu=True,
        num_workers=4,
        )
    )
result: Result = trainer.fit()


data.append(group_res("PolicyNN_23", 1024, 4, result))

config = {"modelName": PolicyNN_23(), "num_epoch":1, "dim_num":23, "label_dim":9, "batch_size": 2048}
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(
        use_gpu=True,
        num_workers=4,
        )
    )
result: Result = trainer.fit()

data.append(group_res("PolicyNN_23", 2048, 4, result))

config = {"modelName": PolicyNN_23(), "num_epoch":1, "dim_num":23, "label_dim":9, "batch_size": 4196}
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(
        use_gpu=True,
        num_workers=4,
        )
    )
result: Result = trainer.fit()

data.append(group_res("PolicyNN_23", 4196, 4, result))

config = {"modelName": PolicyNN_23(), "num_epoch":1, "dim_num":23, "label_dim":9, "batch_size": 8192}
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(
        use_gpu=True,
        num_workers=4,
        )
    )
result: Result = trainer.fit()

data.append(group_res("PolicyNN_23", 8192, 4, result))

config = {"modelName": PolicyNN_23(), "num_epoch":1, "dim_num":23, "label_dim":9, "batch_size": 16384}
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(
        use_gpu=True,
        num_workers=4,
        )
    )
result: Result = trainer.fit()

data.append(group_res("PolicyNN_23", 16384, 4, result))

#8GPU
#PolicyNN_23 with different batch size
config = {"modelName": PolicyNN_23(), "num_epoch":1, "dim_num":23, "label_dim":9, "batch_size": 1024}
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(
        use_gpu=True,
        num_workers=8,
        )
    )
result: Result = trainer.fit()


data.append(group_res("PolicyNN_23", 1024, 8, result))

config = {"modelName": PolicyNN_23(), "num_epoch":1, "dim_num":23, "label_dim":9, "batch_size": 2048}
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(
        use_gpu=True,
        num_workers=8,
        )
    )
result: Result = trainer.fit()

data.append(group_res("PolicyNN_23", 2048, 8, result))

config = {"modelName": PolicyNN_23(), "num_epoch":1, "dim_num":23, "label_dim":9, "batch_size": 4196}
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(
        use_gpu=True,
        num_workers=8,
        )
    )
result: Result = trainer.fit()

data.append(group_res("PolicyNN_23", 4196, 8, result))

config = {"modelName": PolicyNN_23(), "num_epoch":1, "dim_num":23, "label_dim":9, "batch_size": 8192}
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(
        use_gpu=True,
        num_workers=8,
        )
    )
result: Result = trainer.fit()

data.append(group_res("PolicyNN_23", 8192, 8, result))

config = {"modelName": PolicyNN_23(), "num_epoch":1, "dim_num":23, "label_dim":9, "batch_size": 16384}
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(
        use_gpu=True,
        num_workers=8,
        )
    )
result: Result = trainer.fit()

data.append(group_res("PolicyNN_23", 16384, 8, result))





#write the result to the csv file
with open("results.csv", "a", newline="") as csvfile:
    csv_writer= csv.writer(csvfile)
    csv_writer.writerows(data)