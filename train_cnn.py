from cnn_model import CNNDigitModel
from plot import  plot_chart
from constants import learning_rate, weight_decay, device, epochs, custom_format, min_delta, patience
from utils import load_data, get_date_suffix, get_batches
import os
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm



if __name__ == "__main__":
    dataset_train_dir = Path('./dataset/train')
    dataset_test_dir =Path('./dataset/test')
    train_files = list(dataset_train_dir.glob("*.png"))
    test_files = list(dataset_test_dir.glob("*.png"))

    print(f"Loading Training data...")
    train_data = load_data(files=train_files,noisy=True,shape=(1,28,28))
    print(f"Loaded {len(train_data)} train images")
    print(f"Loading Testing data...")
    test_data = load_data(files=test_files,shape=(1,28,28))
    print(f"Loaded {len(test_data)} test images")
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []


    best_test_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    date_suffix = get_date_suffix()
    os.makedirs("./models_cnn", exist_ok=True)
    best_path = f"./models_cnn/model_best_{date_suffix}.pth"
    model = CNNDigitModel()
    model = model.to(device=device)
    optimzer = optim.Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay)

    for i in range(epochs):
      print("Epoch ",i+1)
      x,y,batch = get_batches(data=train_data)
      
      curr_train_acc = []
      total_items = 0
      model.train()
      curr_train_loss = []
      
      for xb,yb in tqdm(zip(x,y),total=len(x),bar_format=custom_format):
        optimzer.zero_grad()
        logits,loss = model(xb,yb)
        total_items += yb.size(0)
        curr_train_acc.append((logits.argmax(dim=1) == yb.argmax(dim=1)).sum().item())
        loss.backward()
        curr_train_loss.append(loss.item())
        optimzer.step()
        
      curr_avg_train_loss = np.average(np.array(curr_train_loss)) 
      curr_avg_train_acc = np.sum(np.array(curr_train_acc)) / total_items
      train_acc.append(curr_avg_train_acc)
      model.eval()
      
      test_x,test_y,batch = get_batches(data=test_data)
      curr_test_loss = []
      curr_test_acc = []
      total_items = 0
      with torch.no_grad():
        for xb,yb in tqdm(zip(test_x,test_y),total=len(test_x),bar_format=custom_format):
          logits,loss = model(xb,yb)
          curr_test_loss.append(loss.item())
          total_items += yb.size(0)
          curr_test_acc.append((logits.argmax(dim=1) == yb.argmax(dim=1)).sum().item())
      curr_avg_test_loss = np.average(np.array(curr_test_loss))
      curr_avg_test_acc = np.sum(np.array(curr_test_acc)) / total_items
      test_acc.append(curr_avg_test_acc)
      train_loss.append(curr_avg_train_loss)
      test_loss.append(curr_avg_test_loss)
      
      print(f"Epoch {i+1} - Train Loss: {curr_avg_train_loss:.4f} - Test Loss: {curr_avg_test_loss:.4f} - Train Acc: {curr_avg_train_acc*100:.2f}% - Test Acc: {curr_avg_test_acc*100:.2f}%")
      if curr_avg_test_loss < best_test_loss - min_delta:
          best_test_loss = curr_avg_test_loss
          best_epoch = i + 1
          epochs_no_improve = 0
          torch.save(model.state_dict(), best_path)
          print(f"  saved best model (test loss {best_test_loss:.4f})")
      else:
          epochs_no_improve += 1
          print(f"  no improvement for {epochs_no_improve}/{patience} epochs")
          if epochs_no_improve >= patience:
              print(f"Early stopping at epoch {i+1}. Best was epoch {best_epoch}")
              break
      
      

    date_suffix = get_date_suffix()
    os.makedirs('./models',exist_ok=True)
    os.makedirs('./plots_cnn/loss',exist_ok=True)
    os.makedirs('./plots_cnn/accuracy',exist_ok=True)
    print(f"Model saved to ./models/model_{date_suffix}.pth")
      
    plot_chart(train_loss, test_loss, 'Epoch', 'Loss', 'Train Loss', 'Test Loss', 'Loss vs Epoch', f'./plots_cnn/loss/loss_{date_suffix}.png')
    plot_chart(train_acc, test_acc, 'Epoch', 'Accuracy', 'Train Accuracy', 'Test Accuracy', 'Accuracy vs Epoch', f'./plots_cnn/accuracy/accuracy_{date_suffix}.png')
    print("Plots saved")
    print("Training complete")