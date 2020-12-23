from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# trainning function
def fit(model, dataloader, optimizer, scheduler, criterion, train_data):
    print('Training')
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    
    for i, (inputs, labels) in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        train_running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_running_correct += (preds==labels).sum().item()
        loss.backward()
        optimizer.step()
        scheduler.step()

    train_loss = train_running_loss / len(dataloader.dataset)
    train_accuracy = 100. * train_running_correct/len(dataloader.dataset)
    return train_loss, train_accuracy

# validation function
def validate(model, dataloader, criterion, val_data):
    print('Validating')
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    with torch.no_grad():
        for i, (inputs, labels) in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            # print(preds, labels)
            val_running_correct += (preds==labels).sum().item()

        val_loss = val_running_loss/len(dataloader.dataset)
        val_accuracy = 100. * val_running_correct/len(dataloader.dataset)
        return val_loss, val_accuracy

