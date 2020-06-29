import numpy
import pickle
import torch

from configparser import ConfigParser
from language_model.language_model import LanguageModel, RelDataset, get_loaders


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device is :" + str(device))
    parser = ConfigParser()
    parser.read('config.txt')
    lang_model = LanguageModel(parser) # load the language model
    lang_model.to(device)
    f_data = parser.get("data", "training_samples")
    with open(f_data, "rb") as f:
        data = pickle.load(f)
    dataset = RelDataset(device, data)
    trainloader, devloader = get_loaders(dataset, batch_size=128)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lang_model.parameters())
    for epoch in range(10):
        print("***** Start of epoch " + str(epoch) + " *****")
        avg_loss = 0.0
        for step, data in enumerate(trainloader):
            lang_model.train()
            optimizer.zero_grad()
            samples = data["samples"]
            targets = torch.tensor(data["targets"], dtype=int)
            outputs = lang_model(samples)
            loss = loss_func(outputs, targets)
            # choices = numpy.argmax(outputs.detach(), axis=1)
            # accuracy = (choices == targets).sum()
            avg_loss += loss
            loss.backward()
            optimizer.step()
        dev_matches = 0.0
        for set, data in enumerate(devloader):
            lang_model.eval()
            samples = data["samples"]
            targets = torch.tensor(data["targets"], dtype=int)
            outputs = lang_model(samples)
            choices = numpy.argmax(outputs.detach(), axis=1)
            dev_matches += (choices == targets).sum()
        print("The average loss on the training set is " + str((avg_loss/len(trainloader)).item()))
        print("Accuracy on the dev set is " + str((dev_matches/len(devloader)).item()) + "%")




