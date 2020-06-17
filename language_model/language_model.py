from flair.embeddings import TransformerWordEmbeddings, FlairEmbeddings, WordEmbeddings, BytePairEmbeddings, \
    CharacterEmbeddings, StackedEmbeddings

class LanguageModel():
    """
    Fine-tune a language model via a binary classifier for identifying semantic language_model
    """
    def __init__(self, config):
        """
        Load pretrained language model
        """
        embeddings_stack = []
        transformers = config.get("language_model", "transformers")
        if transformers is not "":
            transformers = transformers.split(";")
            for code in transformers:
                embeddings_stack.append(TransformerWordEmbeddings(code,
                                                                  layers="all",
                                                                  pooling_operation='mean',
                                                                  use_scalar_mix=True,
                                                                  fine_tune=True))
        word_embeddings = config.get("language_model", "word_embeddings")
        if word_embeddings is not "":
            word_embeddings = word_embeddings.split(";")
            for code in word_embeddings:
                embeddings_stack.append(WordEmbeddings(code))
        flair_embeddings = config.get("language_model", "word_embeddings")
        if flair_embeddings is not "":
            flair_embeddings = flair_embeddings.split(";")
            for code in flair_embeddings:
                embeddings_stack.append(FlairEmbeddings(code))
        char_embeddings = config.get("language_model", "char_embeddings")
        if char_embeddings is not "":
            embeddings_stack.append(CharacterEmbeddings(char_embeddings))
        bytepair_embeddings = config.get("language_model", "bytepair_embeddings")
        if bytepair_embeddings.lower() is "yes":
            embeddings_stack.append(BytePairEmbeddings())
        custom_embeddings = config.get("language_model", "custom_embeddings")
        if custom_embeddings is not "":
            custom_embeddings = custom_embeddings.split(";")
            for path in custom_embeddings:
                embeddings_stack.append(WordEmbeddings(custom_embeddings))
        self.lm = StackedEmbeddings(embeddings_stack)
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