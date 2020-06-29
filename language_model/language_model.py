import numpy
import torch

from flair.data import Sentence
from flair.embeddings import TransformerWordEmbeddings, FlairEmbeddings, WordEmbeddings, BytePairEmbeddings, \
    CharacterEmbeddings, StackedEmbeddings
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler


def get_loaders(dataset, batch_size=128, split=0.01):
    """
    Split the data into train/dev sets and initialize loader objects
    :param dataset:
    :param batch_size:
    :param split:
    :return:
    """
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(numpy.floor(split * dataset_size))
    numpy.random.seed(42)
    numpy.random.shuffle(indices)
    dev_indices, train_indices = indices[:split], indices[split:]
    trainsampler = SubsetRandomSampler(train_indices)
    devsampler = SubsetRandomSampler(dev_indices)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=trainsampler)
    devloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=devsampler)
    return trainloader, devloader

class RelDataset(Dataset):

    def __init__(self, device, data):
        self.device = device
        self.data = data
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ''' Prepare one sample sentence of the data '''
        sample = self.data[idx]
        data = {"samples": sample[0], "targets": sample[1]}
        return data

class LanguageModel(torch.nn.Module):
    """
    Fine-tune a language model via a binary classifier for identifying semantic language_model
    """
    def __init__(self, config):
        """
        Load pretrained language model
        """
        super(LanguageModel, self).__init__()
        embeddings_stack = []
        transformers = config.get("language_model", "transformers")
        if transformers is not "":
            transformers = transformers.split(";")
            for model in transformers:
                embeddings_stack.append(TransformerWordEmbeddings(model,
                                                                  layers="all",
                                                                  pooling_operation='mean',
                                                                  use_scalar_mix=True,
                                                                  fine_tune=True))
        word_embeddings = config.get("language_model", "word_embeddings")
        if word_embeddings is not "":
            word_embeddings = word_embeddings.split(";")
            for model in word_embeddings:
                embeddings_stack.append(WordEmbeddings(model))
        flair_embeddings = config.get("language_model", "flair_embeddings")
        if flair_embeddings is not "":
            flair_embeddings = flair_embeddings.split(";")
            for model in flair_embeddings:
                embeddings_stack.append(FlairEmbeddings(model, fine_tune=True))
        character_embeddings = config.get("language_model", "character_embeddigs")
        if character_embeddings.lower() is "yes":
            embeddings_stack.append(CharacterEmbeddings(character_embeddings))
        bytepair_embeddings = config.get("language_model", "bytepair_embeddings")
        if bytepair_embeddings.lower() is "yes":
            embeddings_stack.append(BytePairEmbeddings())
        custom_embeddings = config.get("language_model", "custom_embeddings")
        if custom_embeddings is not "":
            custom_embeddings = custom_embeddings.split(";")
            for path in custom_embeddings:
                embeddings_stack.append(WordEmbeddings(path))
        self.lm = StackedEmbeddings(embeddings_stack)
        self.embedding_dim = self.lm.embedding_length
        self.classify = torch.nn.Linear(self.embedding_dim, 2)

    def forward(self, data):
        """
        Get contextualized embeddings for input sequence
        :return:
        """
        X = [Sentence(sent) for sent in data]
        self.lm.embed(X)
        # X = [sent[0] for sent in X]
        X = torch.stack([sentence[0].embedding for sentence in X])
        labels = self.classify(X)
        return labels

    def learn_relations(self):
        """
        Fine tune the language model via binary classification of relation identification, i.e. YES or NO answers.
        Data format is:
        [REL_NAME] <gloss of CONCEPT1 or example containing CONCEPT1> [SEP] CONCEPT <gloss of CONCEPT2 or gloss of CONCEPT1>
        e.g. [HYPONYM] feline mammal usually having thick soft fur and no ability to roar: domestic cats; wildcats
             [SEP] cat any of various lithe-bodied roundheaded fissiped mammals, many with retractile claws
        :return:
        """

    def match_relation(self):
        """
        Determine whether an input sequence contains a relation of a particular kind
        :return:
        """

    def id_relation(self):
        """
        Determine whether an input sequence contains a relation out of a set of possible types
        :return:
        """

def accuracy(self):
    """
    Calculate classification accuracy
    :return:
    """

