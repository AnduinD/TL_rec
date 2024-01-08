# main.py
import numpy as np
from torch.utils.data.dataloader import DataLoader

from dataset import TL_dataset, IDX2NAME
from model_raw import MLP



def L1Loss(x,y):
    return np.mean(np.abs(x-y))

def main():
    train_dataset=TL_dataset(idx_list=range(len(IDX2NAME)))
    test_dataset=TL_dataset(idx_list=range(len(IDX2NAME)))

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = MLP(in_channel=9,hidden_channel=3,out_channel=1,alpha=1.)

    criterion = L1Loss
    # optimizer = optim.SGD(model.parameters(), lr=1.)

    num_epochs = 100
    # device = "cpu" #"cuda:0"
    # model.to(device)
    for epoch in range(num_epochs):
        for data, label in train_loader:
            data, label = data.squeeze(0).numpy(),label.squeeze(0).numpy()

            outputs = model.backward(data ,label ,lr=1.)
            # loss = criterion(model.forward(data), label)
            # print('Epoch: #%s, L1Loss: %f' % (epoch, loss))

    # model.eval()
    correct = 0
    total = 0
    # with torch.no_grad():
    for data, label in test_loader:
        data, label = data.squeeze(0).numpy(), label.squeeze(0).numpy()

        outputs = model(data)
        print(outputs, label)
        outputs = outputs > 0.5

        total += label.shape[0]
        correct += (outputs == label).sum().item()

    acc = correct / total
    print("correct:",correct,"total:",total)
    print("acc{:.2%}".format(acc))

    print(f"factor_a: {model.alpha}")
    # np.save("TL_Net.npy",model.state_dict())
    # torch.save(model.state_dict(), "TL_Net.pth")

if __name__ == "__main__":
    main()