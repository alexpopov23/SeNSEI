from configparser import ConfigParser
from language_model.language_model import LanguageModel


if __name__ == "__main__":
    parser = ConfigParser()
    parser.read('config.txt')
    lang_model = LanguageModel(parser) # load the language model
