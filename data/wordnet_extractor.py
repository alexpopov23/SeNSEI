import os
import collections

import xml.etree.ElementTree as ET

from docutils.nodes import term

relation_types = {"hypernym", "hyponym", "entails", "similar to", "meronym member", "holonym member",
                  "meronym substance", "holonym substance", "meronym part", "holonym part", "derivation",
                  "class", "member", "cause", "verb group", "value", "attribute", "antonym", "see also",
                  "participle source", "participle", "pertainym", "derivation", "instance"}


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
                rel_set.add((node1, "hypernym", node2))
                rel_set.add((node2, "hyponym", node1))
            elif reltype == "ent":
                rel_set.add((node1, "entails", node2))
            elif reltype == "sim":
                rel_set.add((node1, "similar to", node2))
                rel_set.add((node2, "similar to", node1))
            elif reltype == "mm":
                rel_set.add((node1, "meronym member", node2))
                rel_set.add((node2, "holonym member", node1))
            elif reltype == "ms":
                rel_set.add((node1, "meronym substance", node2))
                rel_set.add((node2, "holonym substance", node1))
            elif reltype == "mp":
                rel_set.add((node1, "meronym part", node2))
                rel_set.add((node2, "holonym part", node1))
            elif reltype == "der":
                rel_set.add((node1, "derivation", node2))
                rel_set.add((node2, "derivation", node1))
            elif reltype == "cls":
                rel_set.add((node1, "class", node2))
                rel_set.add((node2, "member", node1))
            elif reltype == "cs":
                rel_set.add((node1, "cause", node2))
            elif reltype == "vgp":
                rel_set.add((node1, "verb group", node2))
            elif reltype == "at":
                if pos1 == "n":
                    rel_set.add((node1, "value", node2))
                elif pos1 == "a":
                    rel_set.add((node1, "attribute", node2))
            elif reltype == "ant":
                rel_set.add((node1, "antonym", node2))
                rel_set.add((node2, "antonym", node1))
            elif reltype == "sa":
                rel_set.add((node1, "see also", node2))
            elif reltype == "ppl":
                rel_set.add((node1, "participle source", node2))
                rel_set.add((node2, "participle", node1))
            elif reltype == "per":
                if pos1 == "a":
                    rel_set.add((node1, "pertainym", node2))
                elif pos1 == "r":
                    rel_set.add((node1, "derivation", node2))
                    rel_set.add((node2, "derivation", node1))
            elif reltype == "ins":
                rel_set.add((node2, "instance", node1, True))
    return rel_set

def get_training_samples(rels, terms, lemma2synsets, num_negative=2):
    """
    Construct training samples in the required format:
    "[HYPONYM] feline mammal usually having thick soft fur and no ability to roar: domestic cats; wildcats
     [SEP] cat any of various lithe-bodied roundheaded fissiped mammals, many with retractile claws"
    :return:
    """
    samples = []
    for rel in rels:
        syn1, rel, syn2 = rel[0], rel[1], rel[2]
        term1, term2 = terms[syn1], terms[syn2]
        # Construct a TRUE sample
        # [HYPONYM] / [ANTONYM] might no work, since they are not BERT symbols; won't hurt to try if it can learn them, though
        # sample = "[" + rel + "] " + term1.gloss + " [SEP] " + term2.gloss
        # rel = rel.lower().replace("-", " ") # try some weak supervision
        sample = "[CLS] " + term1.gloss + " [SEP] " + rel + " " + term2.gloss
        samples.append((sample, True))
        # false sample - wrong relation label
        neg_sample = "[CLS] " + term1.gloss + " [SEP] " + random.choice(tuple(relation_types.difference({rel}))) + " " + term2.gloss
        samples.append((neg_sample, False))
        # Construct FALSE samples
        lemmas1 = term1.lemmas
        neg_glosses = set()
        for i in range(2 * num_negative):
            cur_lemma = random.choice(lemmas1).replace(" ", "_").lower()
            cur_syn = random.choice(lemma2synsets[cur_lemma])
            if cur_syn == syn1:
                continue
            cur_gloss = terms[cur_syn].gloss
            if cur_gloss not in neg_glosses:
                neg_glosses.add(cur_gloss)
            if len(neg_glosses) == num_negative:
                break
        if len(neg_glosses) < num_negative:
            extra_glosses = [terms[random_syn].gloss for random_syn in random.sample(list(synset2id), (num_negative - len(neg_glosses)))]
            neg_glosses.update(extra_glosses)
        for neg_gloss in neg_glosses:
            # neg_sample = "[" + rel + "] " + neg_gloss + " [SEP] " + term2.gloss
            neg_sample = "[CLS] " + neg_gloss + " [SEP] " + rel + " " + term2.gloss
            samples.append((neg_sample, False))
        # print("End times")
    return samples

def read_preprocessed_data():
    data_dir = os.path.join(os.getcwd(), "..\data_files\wordnet_data_CLS-WeakSupervision.pkl")
    with open(data_dir, "rb") as f:
        terms = pickle.load(f)
        rels = pickle.load(f)
        lemma2synsets = pickle.load(f)
        lemmapos2synsets = pickle.load(f)
        synset2id = pickle.load(f)
    return terms, rels, lemma2synsets, lemmapos2synsets, synset2id

if __name__ == "__main__":
    # terms = {}
    # cur_dir = os.getcwd()
    # # path2glosses = os.path.join(cur_dir, "..\data_files\wordnet\glosses")
    # path2glosses = "C:\Work\dev\lang-resources\wordnet\WordNet-3.0\glosstag\merged"
    # for suffix in ["adj.xml", "adv.xml", "noun.xml", "verb.xml"]:
    #     path = os.path.join(path2glosses, suffix)
    #     terms.update(get_wordnet_terms(path))
    # # path2rels = os.path.join(cur_dir, "..\data_files\wordnet\\relations")
    # path2rels = "C:\Work\dev\lang-resources\wordnet\\relations\All\\base-relations"
    # rels = set()
    # for f in os.listdir(path2rels):
    #     rels.update(get_wordnet_relations(os.path.join(path2rels, f)))
    # # path2dict = os.path.join(cur_dir, "..\data_files\wordnet\dictionaries\wn30.lex")
    # path2dict = "C:\Work\dev\lang-resources\wordnet\wn30.lex"
    # lemma2synsets, lemmapos2synsets, synset2id = get_wordnet_dict(path2dict)
    # output_f = os.path.join(cur_dir, "..\data_files\wordnet_data_CLS-WeakSupervision.pkl")
    # with open(output_f, "wb") as f:
    #     pickle.dump(terms, f, protocol=2)
    #     pickle.dump(rels, f, protocol=2)
    #     pickle.dump(lemma2synsets, f, protocol=2)
    #     pickle.dump(lemmapos2synsets, f, protocol=2)
    #     pickle.dump(synset2id, f, protocol=2)
    # training_samples = get_training_samples(rels, terms, lemma2synsets)

    terms, rels, lemma2synsets, lemmapos2synsets, synset2id = read_preprocessed_data()
    training_samples = get_training_samples(rels, terms, lemma2synsets)
    corpus_f = "C:\Work\dev\lang-resources\wordnet\\rels_corpus\\wn_rels_1positive2negatives_CLS_WeakSupervision.pkl"
    with open(corpus_f, "wb") as f:
        pickle.dump(training_samples, f, protocol=2)
    print("This is the end (my only friend, the end).")
