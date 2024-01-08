# main.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from dataset import TL_dataset, IDX2NAME, NAME2SAMPLE
from model import TL_Net

def main():
    train_dataset=TL_dataset(idx_list=range(len(IDX2NAME)))
    test_dataset=TL_dataset(idx_list=range(len(IDX2NAME)))

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = TL_Net(in_channel=9,hidden_channel=3,out_channel=1,factor=1.)

    optimizer = optim.Rprop(model.parameters(), lr=1.)

    num_epochs = 100
    device = "cpu" #"cuda:0"
    model.to(device)
    for epoch in range(num_epochs):
        for data, label in train_loader:
            data, label = data.to(device),label.to(device)

            optimizer.zero_grad()
            outputs = model(data)

            loss = torch.abs(outputs-label).mean()
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)

            outputs = model(data)
            print(outputs.detach().numpy(), label.detach().numpy())
            outputs = (outputs > 0.5).float()

            total += label.shape[0]
            correct += (outputs == label).sum().item()

    acc = correct / total
    print("correct:",correct,"total:",total)
    print("acc:{:.2%} ".format(acc))

    print(f"factor_a: {model.act.alpha}")
    print("\nfc1.weight:\n",model.fc1.weight.detach().numpy())
    print("\nfc1.bias:\n",  model.fc1.bias.detach().numpy())
    print("\nfc2.weight:\n",model.fc2.weight.detach().numpy())
    print("\nfc2.bias:\n",  model.fc2.bias.detach().numpy())
    torch.save(model.state_dict(), "TL_Net.pth")

    # print(model(torch.Tensor(NAME2SAMPLE['T_center']).unsqueeze(0).float().flatten(1)))
    # print(model(torch.Tensor(NAME2SAMPLE['L_center']).unsqueeze(0).float().flatten(1)))

if __name__ == "__main__":
    main()