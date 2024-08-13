import os
import torch
from torchvision import datasets, transforms
import torchvision

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

import timm

from tqdm import tqdm
from torchvision import transforms as tt
import torchvision.models as models
import torchmetrics
import argparse

writer = SummaryWriter()

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class QuantizationModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.model = model
        self.dequant = torch.ao.quantization.DeQuantStub()

    
    def forward(self, x):

        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


class Trainer:
    def __init__(self, opt):
        super().__init__()

        self.train_transform = tt.Compose([tt.Resize((opt.image_size,opt.image_size)), tt.RandomCrop(opt.image_size, padding=4,padding_mode='reflect'), 
                                tt.RandomHorizontalFlip(), 
                                tt.ToTensor(), 
                                tt.Normalize(MEAN,STD,inplace=True)])

        self.val_transform = tt.Compose([tt.Resize((opt.image_size,opt.image_size)), tt.ToTensor(), tt.Normalize(MEAN,STD)])


        if opt.dataset == "pets37":
            dataset = torchvision.datasets.OxfordIIITPet
            train_dataset = dataset("./",  split="trainval", transform=self.train_transform, download=True)
            val_dataset = dataset("./",  split="test", transform=self.val_transform, download=True)
            num_classes = 37
        else:
            ## Change here
            train_dataset = None
            val_dataset = None
            num_classes = 2

        if opt.weights is not None:
            model = torch.load(opt.weights).to("cpu")

        q_model = QuantizationModel(model)
        q_model.eval()

        if opt.target_device == "server":
            q_model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
        else:
            q_model.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')

        # q_model_fused = torch.ao.quantization.fuse_modules(q_model,
        #     [['conv', 'bn', 'relu']])

        model_prepared = torch.ao.quantization.prepare_qat(q_model.train())

        self.model = model_prepared.to(opt.device)

        self.model_optimizer = optim.Adam(self.model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, 15, 0.1)

        self.ce = nn.CrossEntropyLoss()

        self.train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, drop_last=False)
        self.val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers, drop_last=False)        

        self.criterion = self.ce
        self.epochs = opt.epochs
        self.device = opt.device
        if opt.model_name is None:
            self.model_name = opt.weights.replace(".pt","")
        else:
            self.model_name = opt.model_name



    def accuracy(self, outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))


    def process_batch(self, batch):
            inp, label = batch
            inp = inp.to(self.device)
            label = label.to(self.device)

            pred_t = self.model(inp)

            loss = self.criterion(pred_t, label)

            return loss, pred_t


    def train(self, epoch):
        self.model.train()
        train_loss = 0

        with tqdm(self.train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            for idx, batch in enumerate(tepoch):
                self.model_optimizer.zero_grad()

                loss, pred = self.process_batch(batch)
                loss.backward()
                self.model_optimizer.step()
                train_loss += loss.item()

                tepoch.set_postfix(loss=train_loss/(idx+1))
                writer.add_scalar("RunningLoss/train", loss.item(), len(self.train_loader)*epoch + idx)

        writer.add_scalar("Loss/train", train_loss/(idx+1), epoch)


    def val(self, epoch):    
        self.model.eval()
        preds = []
        labels = []        
        val_loss = 0

        with tqdm(self.val_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Val epoch {epoch}")

            for idx, batch in enumerate(tepoch):
                _, label = batch
                
                with torch.no_grad():
                    loss, pred = self.process_batch(batch)
                preds.append(pred.detach().cpu().argmax(-1))
                labels.append(label)
                val_loss += loss.item()

                tepoch.set_postfix(val_loss=val_loss/(idx+1))

            preds = torch.ravel(torch.cat(preds, dim=0))
            labels = torch.ravel(torch.cat(labels, dim=0))

            acc = torch.tensor(torch.sum(preds == labels).item() / len(preds))
            print(f"Accuracy is: {acc}")

        writer.add_scalar("Loss/val", val_loss/(idx+1), epoch)
        writer.add_scalar("Metric/Acc", round(acc.item(),3), epoch)

        return acc


    def train_eval(self):

        best_acc = 0
        saved_model = None
        
        for epoch in range(self.epochs):
            print(f'Epoch {epoch}/{self.epochs - 1}')

            self.train(epoch)
            self.model_lr_scheduler.step()
        
            acc = self.val(epoch)

            if acc.mean()>best_acc:
                print("found best model")

                if saved_model is not None:
                    os.remove(saved_model)
                saved_model = f"quantized_{self.model_name}_{acc.item()}.pt"

                best_acc = acc.item()
                cpu_mod = self.model.to("cpu")
                model_int8 = torch.ao.quantization.convert(cpu_mod)

                torch.save(model_int8, f"quantized_{self.model_name}_{acc.item()}.pt")

                self.model.to("cuda:0")
            

def parse_args():
    parser = argparse.ArgumentParser()
    list_of_devices = [-1, 0, 1, 2, 3]
    list_of_models = timm.list_models()
    list_of_datasets = ["pets37", "other"]
    target_devices = ["server", "embedded"]


    parser.add_argument('--device', type=int, help='cuda device, i.e. 0 or 0,1,2,3 or cpu (-1)', default=0, choices=list_of_devices)
    parser.add_argument('--dataset', type=str, default='pets37', help='Your Training Dataset', choices=list_of_datasets)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128, help='total batch size for all GPUs')

    parser.add_argument("--image-size", type=int, help="Training height", default=224)
    parser.add_argument("--workers", type=int, help="Dataloader Workers", default=8)
    parser.add_argument("--lr", type=int, help="Learning Rate", default=1e-3)
    parser.add_argument('--weights', type=str, default=None, required=True, help='Timm Used Model')

    parser.add_argument('--model-name', type=str, default=None, help='Your Model Name')

    parser.add_argument('--target-device', type=str, help='Target Device', choices=target_devices)

    parser.add_argument("--weight-decay", type=float, help="Weight Decay", default=0.001)


    return parser.parse_args()







if __name__ == '__main__':
    opt = parse_args()    

    trainer = Trainer(opt)
    trainer.train_eval()