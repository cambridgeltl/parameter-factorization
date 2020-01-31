""" Load sequential dataset. """

from __future__ import absolute_import, division, print_function

import logging
from io import open
import numpy as np
from sklearn.metrics import f1_score
import codecs

logger = logging.getLogger(__name__)
utf8reader = codecs.getreader('utf-8')

CLASSES_PER_TASK = {'ner': ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-GPE', 'I-GPE', 'B-LOC', 'I-LOC'],
                    'pos': ['ADJ', 'ADP', 'PUNCT', 'ADV', 'AUX', 'SYM', 'INTJ', 'CCONJ', 'X', 'NOUN', 'DET',
                            'PROPN', 'NUM', 'VERB', 'PART', 'PRON', 'SCONJ'],
                    'nli': ['contradiction', 'neutral', 'entailment']
                    }


class CoNLLExample(object):
    """
    A single training/test example for a CoNLL dataset.
    """

    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = "tokens: '%s'" % (" ".join(self.tokens))
        s += ", labels: %s" % (" ".join(self.labels))
        return s


class XNLIExample(object):
    """
    A single training/test example for the XNLI dataset.
    """

    def __init__(self, text_a, text_b, label):
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = "premise: '%s'" % (" ".join(self.text_a))
        s += ", hypo: %s" % (" ".join(self.text_b))
        s += ", label: %s" % (self.label)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_lengths, label_ids):
        self.input_ids = input_ids
        self.input_lengths = input_lengths
        self.label_ids = label_ids


def read_conll_examples(input_file, task):
    """Read a CoNLL file into a list of CoNLLExample."""
    if task == "ner":
        token_col, label_col = 0, -1
        separator = ' '
    elif task == "pos":
        token_col, label_col = 1, 3
        separator = '\t'
    else:
        raise NotImplementedError
    reader = open(input_file, "r", encoding='utf-8')

    examples = []
    tokens, labels = [], []

    for line in reader:
        if not line.strip():
            example = CoNLLExample(
                tokens=tokens,
                labels=labels,
                )
            examples.append(example)
            tokens, labels = [], []
        elif task == 'pos' and (line.startswith('#') or '-' in line.split(separator)[0] or '.' in line.split(separator)[0]):
            continue
        else:
            cols = line.strip().split(separator)
            assert len(cols) >= 2, cols
            token, label = cols[token_col], cols[label_col]
            tokens.append(token)
            labels.append(label)

    if len(tokens) > 0 and len(labels) > 0:
        example = CoNLLExample(
            tokens=tokens,
            labels=labels,
            )
        examples.append(example)

    return examples


def read_xnli_examples(input_file):
    examples = []
    fin = open(input_file)
    for line in fin:
        premise, hypo, label = line.strip().split("\t")
        examples.append(
            XNLIExample(text_a=premise.split(), text_b=hypo.split(), label=label))
    return examples


def convert_examples_to_features(examples, label_map, tokenizer,
                                 max_seq_length, pad_token=0, mask_padding_with_zero=True):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    CLS = "[CLS]"
    SEP = "[SEP]"
    for (example_index, example) in enumerate(examples):

        if example_index % 1000 == 0:
            logger.info('Converting %s/%s', example_index, len(examples))

        tokens = example.tokens
        orig_to_tok_index = np.zeros(max_seq_length)
        sub_tokens = [CLS]
        for token in tokens:
            wordpieces = tokenizer.tokenize(token)
            if len(sub_tokens) + len(wordpieces) > max_seq_length-1:
                break
            orig_to_tok_index[len(sub_tokens)] = 1
            sub_tokens.extend(wordpieces)
        sub_tokens += [SEP]

        input_ids = tokenizer.convert_tokens_to_ids(sub_tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_length = len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - input_length
        input_ids = input_ids + ([pad_token] * padding_length)

        assert len(input_ids) == max_seq_length

        label_ids = np.array([-1] * max_seq_length, np.int8)
        labels = example.labels[:int(orig_to_tok_index.sum())]
        labels = [label_map[l] for l in labels]
        label_ids[np.array(orig_to_tok_index) > 0] = labels

        assert len(label_ids) == max_seq_length

        if example_index < 5:
            logger.info("*** Example ***")
            logger.info("id: %s" % (example_index))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("labels: %s" % " ".join(
                    [str(x) for x in labels]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("length: %s" % str(input_length))
            logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_lengths=[input_length],
                              label_ids=label_ids))

    return features


def glue_convert_examples_to_features(examples, label_map, tokenizer,
                                      max_seq_length, pad_token=0, mask_padding_with_zero=True,
                                      pad_token_segment_id=0):

    features = []
    CLS = "[CLS]"
    SEP = "[SEP]"
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))
        inputs = [CLS]
        for token in example.text_a:
            subtoken = tokenizer.tokenize(token)
            inputs += subtoken
        inputs += [SEP]
        lengths = [min(len(inputs), max_seq_length)]
        for token in example.text_b:
            subtoken = tokenizer.tokenize(token)
            inputs += subtoken
        inputs += [SEP]
        lengths += [min(len(inputs), max_seq_length)]
        input_ids = tokenizer.convert_tokens_to_ids(inputs)
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)

        input_ids = input_ids + ([pad_token] * padding_length)

        assert len(input_ids) == max_seq_length, "Error with input length {} vs {}".format(len(input_ids), max_seq_length)

        label = 'contradiction' if example.label == 'contradictory' else example.label
        label = label_map[label]

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (ex_index))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_length: %s" % " ".join([str(x) for x in lengths]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(InputFeatures(input_ids=input_ids,
                                      input_lengths=lengths,
                                      label_ids=[label]))

    return features


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    acc = simple_accuracy(preds, labels)
    labels_to_include = range(1 if task_name == 'ner' else 0, len(CLASSES_PER_TASK[task_name]))
    f1 = f1_score(y_true=labels, y_pred=preds, average='micro', labels=labels_to_include)
    return {
        "acc": acc,
        "f1": f1,
           }
