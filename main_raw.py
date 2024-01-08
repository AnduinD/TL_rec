# main.py
import numpy as np
from torch.utils.data.dataloader import DataLoader

from dataset import TL_dataset, IDX2NAME
from model_raw import MLP

def main():
    train_dataset=TL_dataset(idx_list=range(len(IDX2NAME)))
    test_dataset= TL_dataset(idx_list=range(len(IDX2NAME)))

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = MLP(in_channel=9,hidden_channel=3,out_channel=1,alpha=1.)

    num_epochs = 1000
    for _ in range(num_epochs):
        for data, label in train_loader:
            data, label = data.squeeze(0).numpy(),label.squeeze(0).numpy()
            outputs = model.backward(data ,label ,lr=1.)

        err=[]
        for data, label in test_loader:
            data, label = data.squeeze(0).numpy(), label.squeeze(0).numpy()
            err.append(model(data)-label)
        
        if np.sum(np.abs(err)) < 0.1:
            break
        

    correct = 0
    total = 0
    err=[]
    for data, label in test_loader:
        data, label = data.squeeze(0).numpy(), label.squeeze(0).numpy()

        outputs = model(data)
        err.append(outputs-label)
        print(outputs, label)
        outputs = outputs > 0.5

        total += label.shape[0]
        correct += (outputs == label).sum().item()

    acc = correct / total
    print("correct:",correct,"total:",total)
    print("acc: {:.2%}".format(acc))
    print("err: {:.2}".format(np.sum(np.abs(err))))

    print(f"factor_a: {model.alpha}")

    print("\nunit1.weight:\n",model.unit1.weight)
    print("\nunit1.bias:\n",  model.unit1.bias)
    print("\nunit2.weight:\n",model.unit2.weight)
    print("\nunit2.bias:\n",  model.unit2.bias)


if __name__ == "__main__":
    main()