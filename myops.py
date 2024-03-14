import torch 
import tqdm
from torch.cuda.amp import autocast
import numpy as np

def Train_FT_MTSL_MTHD_V2(epoch, dataloader, model, criterion, optimizer, scheduler, scaler, device, writer):
    # regular cross-entropy loss training
    model.train()
    running_loss = 0.0
    
    # for accuracy
    total = 0
    correct = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0    
    
    i = 0
    for batch in tqdm.tqdm(dataloader):
        # Move the data to the device
        images = batch[0]
        images = [images[0].to(device), images[1].to(device), images[2].to(device)]
        labels = batch[2].to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Perform the forward pass for mixed precision training
        with autocast():
            outputs = model(images)
            logit, logit1, logit2, logit3, ft, ft1, ft2, ft3 = outputs
            
            loss = criterion(logit, labels)
            loss1 = criterion(logit1, labels)
            loss2 = criterion(logit2, labels)
            loss3 = criterion(logit3, labels)
            
            loss = loss + loss1 + loss2 + loss3

        # Perform the backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update the learning rate
        scheduler.step()

        # Update the running loss
        running_loss += loss.item() * images[0].size(0)

        # update the running accuracy        
        total += labels.size(0)

        # Prediction for each modality
        _, predicted1 = torch.max(logit1.data, 1)
        _, predicted2 = torch.max(logit2.data, 1)
        _, predicted3 = torch.max(logit3.data, 1)
        correct1 += (predicted1 == labels).sum().item()
        correct2 += (predicted2 == labels).sum().item()
        correct3 += (predicted3 == labels).sum().item()
        
        # overall prediction
        _, predicted = torch.max(logit.data, 1)
        correct += (predicted == labels).sum().item()
    

        # Log training loss to TensorBoard
        writer.add_scalar('Train/Running/Loss', loss.item(), epoch*len(dataloader)+i)
        writer.add_scalar('Train/Running/Loss1', loss1.item(), epoch*len(dataloader)+i)
        writer.add_scalar('Train/Running/Loss2', loss2.item(), epoch*len(dataloader)+i)
        writer.add_scalar('Train/Running/Loss3', loss3.item(), epoch*len(dataloader)+i)
        
        writer.add_scalar('Train/Running/Acc', correct / total, epoch*len(dataloader)+i)
        writer.add_scalar('Train/Running/Acc1', correct1 / total, epoch*len(dataloader)+i)
        writer.add_scalar('Train/Running/Acc2', correct2 / total, epoch*len(dataloader)+i)
        writer.add_scalar('Train/Running/Acc3', correct3 / total, epoch*len(dataloader)+i)
        
        writer.add_scalar('Train/Running/LR', optimizer.param_groups[0]['lr'], epoch*len(dataloader)+i)
                            
        i += 1   
        
                          
    # Calculate the average loss
    epoch_loss = running_loss / len(dataloader.dataset)
    # Calculate the accuracy
    accuracy_overall = correct / total
    accuracy_overall_1 = correct1 / total
    accuracy_overall_2 = correct2 / total
    accuracy_overall_3 = correct3 / total
    
    writer.add_scalar('Train/Epoch/Loss', epoch_loss, epoch)
    writer.add_scalar('Train/Epoch/Acc', accuracy_overall, epoch)
    writer.add_scalar('Train/Epoch/Acc1', accuracy_overall_1, epoch)
    writer.add_scalar('Train/Epoch/Acc2', accuracy_overall_2, epoch)
    writer.add_scalar('Train/Epoch/Acc3', accuracy_overall_3, epoch)
    
    return model, epoch_loss, accuracy_overall


def Test_FT_MTSL_MTHD_V2(epoch, test_loader, model, criterion, device, writer, mode="Val"): 
    # mode ["Val", "Test"]
    model.eval()
    running_loss = 0.0
    
    # for accuracy
    total = 0
    correct = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0

    i=0
    
    for batch in tqdm.tqdm(test_loader):
        # Move the data to the device
        images = batch[0]
        images = [images[0].to(device), images[1].to(device), images[2].to(device)]
        labels = batch[2].to(device)

        # not necessary, but help reduce memory usage
        with torch.no_grad():
            outputs = model(images)
            logit, logit1, logit2, logit3, ft, ft1, ft2, ft3 = outputs
            
            loss = criterion(logit, labels)
            loss1 = criterion(logit1, labels)
            loss2 = criterion(logit2, labels)
            loss3 = criterion(logit3, labels)
            
            loss = loss + loss1 + loss2 + loss3

        # Update the running loss
        running_loss += loss.item() * images[0].size(0)

        # update the running accuracy        
        total += labels.size(0)
        
        # Prediction for each modality
        _, predicted1 = torch.max(logit1.data, 1)
        _, predicted2 = torch.max(logit2.data, 1)
        _, predicted3 = torch.max(logit3.data, 1)
        correct1 += (predicted1 == labels).sum().item()
        correct2 += (predicted2 == labels).sum().item()
        correct3 += (predicted3 == labels).sum().item()
        
        # overall prediction
        _, predicted = torch.max(logit.data, 1)
        correct += (predicted == labels).sum().item()
        
        i += 1       
        
                                
    # Calculate the average loss
    epoch_loss = running_loss / len(test_loader.dataset)
    # Calculate the accuracy
    accuracy_overall = correct / total
    accuracy_overall_1 = correct1 / total
    accuracy_overall_2 = correct2 / total
    accuracy_overall_3 = correct3 / total
    
    writer.add_scalar(mode+'/Epoch/Loss', epoch_loss, epoch)
    writer.add_scalar(mode+'/Epoch/Acc', accuracy_overall, epoch)
    writer.add_scalar(mode+'/Epoch/Acc1', accuracy_overall_1, epoch)
    writer.add_scalar(mode+'/Epoch/Acc2', accuracy_overall_2, epoch)
    writer.add_scalar(mode+'/Epoch/Acc3', accuracy_overall_3, epoch)
    
    return epoch_loss, accuracy_overall