import pandas as pd
import torch
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
from tqdm import tqdm
from .contrastive_loss import *

def train(model, data_loader, train_optimizer, epoch, epochs, batch_size=32, temperature=0.5, device='cuda'):
    """Trains the model defined in ./model.py with one epoch.
    
    Inputs:
    - model: Model class object as defined in ./model.py.
    - data_loader: torch.utils.data.DataLoader object; loads in training data. You can assume the loaded data has been augmented.
    - train_optimizer: torch.optim.Optimizer object; applies an optimizer to training.
    - epoch: integer; current epoch number.
    - epochs: integer; total number of epochs.
    - batch_size: Number of training samples per batch.
    - temperature: float; temperature (tau) parameter used in simclr_loss_vectorized.
    - device: the device name to define torch tensors.

    Returns:
    - The average loss.
    """
    model.train()#设置model在训练模式，与model.eval()相对于
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for data_pair in train_bar:
        x_i, x_j, target = data_pair
        x_i, x_j = x_i.to(device), x_j.to(device)
        
        out_left, out_right, loss = None, None, None
        ##############################################################################
        # TODO: Start of your code.                                                  #
        #                                                                            #
        # Take a look at the model.py file to understand the model's input and output.
        # Run x_i and x_j through the model to get out_left, out_right.              #
        # Then compute the loss using simclr_loss_vectorized.                        #
        ##############################################################################
        _,out_left=model(x_i)
        _,out_right=model(x_j)
        loss=simclr_loss_vectorized(out_left,out_right,temperature)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


def train_val(model, data_loader, train_optimizer, epoch, epochs, device='cuda'):
    is_train = train_optimizer is not None
    model.train() if is_train else model.eval()
    loss_criterion = torch.nn.CrossEntropyLoss()

    total_loss, total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in data_bar:
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%'
                                     .format('Train' if is_train else 'Test', epoch, epochs, total_loss / total_num,
                                             total_correct_1 / total_num * 100, total_correct_5 / total_num * 100))

    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100


def test(model, memory_data_loader, test_data_loader, epoch, epochs, c, temperature=0.5, k=200, device='cuda'):
    model.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        # for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
        
        for data, _, target in memory_data_loader:
            feature, out = model(data.to(device))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()#，cat后转置，并改变了张量在内存中的存储顺序
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            #取前200个分数最高的，然后对同一类的相似度进行求和，取出前五类
            data, target = data.to(device), target.to(device)
            feature, out = model(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            #feature:[B,D]
            #feature_bank:[D,N]
            #计算test_batch和训练集的特征池相似度
            sim_matrix = torch.mm(feature, feature_bank)
            
            # [B, K]
            #挑选前k个相似
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            #feature_labels.expand(data.size(0), -1)得到[B,N]，意义是训练集的N个标签重复Batch次，根据sim_indices挑选
            #好好理解torch.gather的方法，此处sim_indicecs为二维[B,K]，由于dim=-1表示最后一维，所以会从feature_labels中挑选出...，下面这个例子
            #比如sim_indices[3,4]=18,所以会从扩展玩的feature_labels中挑选[3,18]，也就是说第二维的4变成18
            #
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=device)
            # [B*K, C]
            #将index所在的类别置1，其余为0，B*K是为了说明每个测试样本对应K个最相关训练图片
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            #将one_hot_label给reshape回[batch,K,c]，sim_weight给升维到[batch,K,1]，然后再sum相加得到[batch,c]表示每个类别的分数
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            
            pred_labels = pred_scores.argsort(dim=-1, descending=True)#参数是否降序
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100
