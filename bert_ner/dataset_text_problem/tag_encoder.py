import re
import json
import six

import tensorflow as tf

class TagEncoder(object):

    def __init__(self, vocab_filename=None):
        self._init_vocab_from_file(vocab_filename)

    def encode(self, tag):
        return self._token_to_id.get(tag, len(self._token_to_id))

    def decode(self, ids):
        return self._id_to_token.get(ids, "None")

    def _init_vocab(self, token_generator):
        self._id_to_token = {}
        non_reserved_start_index = 0

        self._id_to_token.update(
            enumerate(token_generator, start=non_reserved_start_index)
        )

        self._token_to_id = dict(
            (v, k) for k, v in six.iteritems(self._id_to_token)
        )

    def _init_vocab_from_file(self, filename):
        with tf.gfile.Open(filename) as f:
            tokens = [token.strip() for token in f.readlines()]

        def token_gen():
            for token in tokens:
                yield token
        return self._init_vocab(token_gen())


if __name__ == '__main__':
    vocab_filename = "/path.to/vocab"
    tag_encoder = TagEncoder(vocab_filename)
    label = "CAD:before_DCT:mention:not_mentioned:not_mentioned"
    label_id = tag_encoder.encode(label)
    print(label_id)
    print(tag_encoder.decode(2))
