import torch
from ray.air import session, Checkpoint, ScalingConfig
from ray import train
from models import PolicyNN_60, PolicyNN_48, PolicyNN_24, PolicyNN_23, PolicyNN_108, PolicyNN_211
from ray.train.torch import TorchTrainer



def train_func(config):
    model = train.torch.prepare_model(config["modelName"])
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    
    input = torch.randn(1000, 60)
    labels = torch.randn(1000, 8)
    dataset = torch.utils.data.TensorDataset(input, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
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
        
        
        
    
    
    



config = {"modelName": PolicyNN_60(), "num_epoch":1}
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(
        use_gpu=True,
        num_workers=1,
        )
    )
result = trainer.fit() 
print(result.checkpoint.to_dict())
    
    