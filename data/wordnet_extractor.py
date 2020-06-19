import os
import collections

import xml.etree.ElementTree as ET

class WordNetTerm():
    """
    Structure that holds information about separate WordNet definitions
    """
    def __init__(self):
        self.synset = ""
        self.senses = []
        self.lemmas = []
        self.gloss = ""

def get_wordnet_dict(path):
    """
    Reads the WordNet dictionary
    Args:
        path: A string, the path to the dictionary
    Returns:
        lemma2synsets: A dictionary, maps lemmas to synset IDs
    """
    lemma2synsets, lemmapos2synsets, synset2id, id = {}, {}, {}, 0
    with open(path, "r") as f:
        for line in f.readlines():
            fields = line.split(" ")
            lemma, synsets = fields[0], fields[1:]
            for i, entry in enumerate(synsets):
                synset = entry[:10].strip()
                if synset not in synset2id:
                    synset2id[synset] = id
                    id += 1
                if lemma not in lemma2synsets:
                    lemma2synsets[lemma] = [synset]
                else:
                    lemma2synsets[lemma].append(synset)
                lemmapos = lemma + "-" + synset[-1]
                if lemmapos not in lemmapos2synsets:
                    lemmapos2synsets[lemmapos] = [synset]
                else:
                    lemmapos2synsets[lemmapos].append(synset)
    lemma2synsets = collections.OrderedDict(sorted(lemma2synsets.items()))
    lemmapos2synsets = collections.OrderedDict(sorted(lemmapos2synsets.items()))
    return lemma2synsets, lemmapos2synsets, synset2id

def get_wordnet_terms(path):
    """
    Extract a dictionary mapping synset/senses to glosses
    :param path:
    :return:
    """
    terms = {} # map senses & synsets to WordNetTerm objects
    synsets = ET.parse(path).getroot().findall("synset")
    for synset in synsets:
        term = WordNetTerm()
        syn = synset.get("id")
        term.synset = syn[1:] + "-" + syn[0]
        term.senses = [sense.text for sense in synset.find("keys").findall("sk")]
        term.lemmas = [lemma.text for lemma in synset.find("terms").findall("term")]
        for gloss in synset.findall("gloss"):
            if gloss.get("desc") == "orig":
                term.gloss = gloss.find("orig").text
                break
        terms[term.synset] = term
        for sense in term.senses:
            terms[sense] = term
    return terms

def get_wordnet_relations(path):
    """
    Extract relational knowledge from WordNet.
    Dictionary with relation types as keys and values -- dictionaries that map senses/sysets to senses/synsets.
    :param path:
    :return:
    """
    rel_set = set()
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            fields = line.split()
            node1, node2, reltype, pos1, pos2 = fields[0][2:], fields[1][2:], fields[2][4:], fields[0][-1], fields[1][-1]
            if reltype == "hyp":
                rel_set.add((node1, "HYPERNYM", node2))
                rel_set.add((node2, "HYPONYM", node1))
            elif reltype == "ent":
                rel_set.add((node1, "ENTAILS", node2))
            elif reltype == "sim":
                rel_set.add((node1, "SIMILARITY", node2))
                rel_set.add((node2, "SIMILARITY", node1))
            elif reltype == "mm":
                rel_set.add((node1, "MERONYM-MEMBER", node2))
                rel_set.add((node2, "HOLONYM-MEMBER", node1))
            elif reltype == "ms":
                rel_set.add((node1, "MERONYM-SUBSTANCE", node2))
                rel_set.add((node2, "HOLONYM-SUBSTANCE", node1))
            elif reltype == "mp":
                rel_set.add((node1, "MERONYM-PART", node2))
                rel_set.add((node2, "HOLONYM-PART", node1))
            elif reltype == "der":
                rel_set.add((node1, "DERIVATION", node2))
                rel_set.add((node2, "DERIVATION", node1))
            elif reltype == "cls":
                rel_set.add((node1, "CLASS", node2))
                rel_set.add((node2, "MEMBER", node1))
            elif reltype == "cs":
                rel_set.add((node1, "CAUSE", node2))
            elif reltype == "vgp":
                rel_set.add((node1, "VERB-GROUP", node2))
            elif reltype == "at":
                if pos1 == "n":
                    rel_set.add((node1, "VALUE", node2))
                elif pos1 == "a":
                    rel_set.add((node1, "ATTRIBUTE", node2))
            elif reltype == "ant":
                rel_set.add((node1, "ANTONYM", node2))
                rel_set.add((node2, "ANTONYM", node1))
            elif reltype == "sa":
                rel_set.add((node1, "SEE-ALSO", node2))
            elif reltype == "ppl":
                rel_set.add((node1, "PARTICIPLE-SOURCE", node2))
                rel_set.add((node2, "PARTICIPLE", node1))
            elif reltype == "per":
                if pos1 == "a":
                    rel_set.add((node1, "PERTAINYM", node2))
                elif pos1 == "r":
                    rel_set.add((node1, "DERIVATION", node2))
                    rel_set.add((node2, "DERIVATION", node1))
            elif reltype == "ins":
                rel_set.add((node2, "INSTANCE", node1, True))
    return rel_set

def get_training_samples(rels, terms, lemma2synsets):
    """
    Construct training samples in the required format:
    "[HYPONYM] feline mammal usually having thick soft fur and no ability to roar: domestic cats; wildcats
     [SEP] cat any of various lithe-bodied roundheaded fissiped mammals, many with retractile claws"
    :return:
    """
    for rel in rels:
        node1, rel, node2 = rel[0], rel[1], rel[2]

    return

if __name__ == "__main__":
    terms = {}
    cur_dir = os.getcwd()
    path2glosses = os.path.join(cur_dir, "..\data_files\wordnet\glosses")
    for suffix in ["adj.xml", "adv.xml", "noun.xml", "verb.xml"]:
        path = os.path.join(path2glosses, suffix)
        terms.update(get_wordnet_terms(path))
    path2rels = os.path.join(cur_dir, "..\data_files\wordnet\\relations")
    rels = set()
    for f in os.listdir(path2rels):
        rels.update(get_wordnet_relations(os.path.join(path2rels, f)))
    path2dict = os.path.join(cur_dir, "..\data_files\wordnet\dictionaries\wn30.lex")
    lemma2synsets, lemmapos2synsets, synset2id = get_wordnet_dict(path2dict)
    training_samples = get_training_samples(rels, terms, lemma2synsets)
    print("This is the end (my only friend, the end).")
