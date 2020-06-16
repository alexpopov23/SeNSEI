

class LanguageModel():
    """
    Fine-tune a language model via a binary classifier for identifying semantic language_model
    """
    def __init__(self):
        """
        Load pretrained language model
        """
        return

    def __call__(self):
        """
        Get contextualized embeddings for input sequence
        :return:
        """

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