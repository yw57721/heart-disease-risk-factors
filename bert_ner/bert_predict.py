import os

import collections
import tensorflow as tf
import json
from absl import flags, app
from pprint import pprint

from bert import tokenization, run_classifier, modeling
import bert

# from i2b2.text_multi_tasks import MultiTaskClassification
# from i2b2.dataset_multi_tasks import datasets
from bert_ner.dataset_text_problem.xml_parser import XmlParser
# from third_party import tokenization
from bert_ner.bert_text import model_fn
from bert_ner.dataset_text_problem import tag_encoder
from bert_ner.write2xml import write2xml

FLAGS = flags.FLAGS

# # vocabulary
flags.DEFINE_string("xml_file", os.path.expanduser("~/data/hsu/i2b2/EMR/testing-RiskFactors-Gold/213-044.xml"),
                    "Path of xml file.")
flags.DEFINE_string("xml_path", os.path.expanduser("~/data/hsu/i2b2/testing-RiskFactors-Gold"),
                    "Path of xml files.")
flags.DEFINE_string("export_xml_path", os.path.expanduser("~/data/hsu/i2b2/EMR/testing-RiskFactors-Predicted"),
                    "Path of xml files.")
flags.DEFINE_string("text_to_multi_labels", os.path.expanduser("~/data/hsu/i2b2/train_n_dev/text_to_multi_labels.json"),
                    "Path of text to labels mappings.")

xml_parser = XmlParser()
# subword_tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file)
# tags_decoder = TagEncoder()


def read_xml_text(xmlpath):
    sent_list = xml_parser.process_one(xmlpath, text_only=True)
    return sent_list


def read_text(xml_files):
    # xml_files = [file for file in os.listdir(xml_path) if file.endswith(".xml")]
    # xml_files = [os.path.join(xml_path, file) for file in xml_files]
    all_sent = []
    all_sent_length = []
    for xml_file in xml_files:
        sent = read_xml_text(xml_file)
        num_sent = len(sent)
        all_sent.extend(sent)
        all_sent_length.append(num_sent)

    return all_sent, all_sent_length


def predict_by_rule(input_sentences, text_to_multi_labels):
    predictions = []
    for sent in input_sentences:
        labels = [None]
        for text in text_to_multi_labels:
            if text in sent:
                labels = text_to_multi_labels[text]
        predictions.append(labels)
    return predictions


def main(_):
    params = {}
    for k in FLAGS:
        params[k] = FLAGS[k].value

    params["batch_size"] = 32
    params["init_checkpoint"] = os.path.join(params["pretrained_bert"], "bert_model.ckpt")

    # tokenizer
    tokenizer = bert.tokenization.FullTokenizer(vocab_file=os.path.join(params["pretrained_bert"], "vocab.txt"),
                                                do_lower_case=True)
    params["bert_config"] = modeling.BertConfig.from_json_file(
        os.path.join(params["pretrained_bert"], "bert_config.json"))

    with open(params["label_list"]) as f:
        label_list = f.read().split("\n")
    params["num_labels"] = len(label_list)

    run_config = tf.estimator.RunConfig(
        model_dir=params["model_dir"]
    )
    estimator = tf.estimator.Estimator(
        model_fn=model_fn, params=params, config=run_config)

    label_encoder = tag_encoder.TagEncoder(vocab_filename=params["label_list"])

    if not os.path.isdir(params["export_xml_path"]):
        os.mkdir(params["export_xml_path"])

    xmlpath = params["xml_file"]
    if os.path.isfile(xmlpath):
        xml_files = [xmlpath]
    else:
        xml_files = [file for file in os.listdir(params["xml_path"]) if file.endswith(".xml")]
        xml_files = sorted(xml_files)
        xml_files = [os.path.join(params["xml_path"], file) for file in xml_files]

    # read all sentences from xml files
    input_sentences, input_sentence_lengths = read_text(xml_files)

    # predict
    input_examples = [bert.run_classifier.InputExample(
        guid="",
        text_a=x, text_b=None, label="null:null:null:null:null"
    ) for x in input_sentences]
    input_features = bert.run_classifier.convert_examples_to_features(
        input_examples, label_list,
        params["max_sequence_length"],
        tokenizer
    )
    predict_input_fn = bert.run_classifier.input_fn_builder(
        features=input_features,
        seq_length=params["max_sequence_length"],
        is_training=False,
        drop_remainder=False
    )
    # prediction by MT
    model_predictions = []
    pred_ids = estimator.predict(input_fn=predict_input_fn)
    for pred in pred_ids:
        label = label_encoder.decode(pred)
        model_predictions.append(label)

    # predict by rule
    if os.path.isfile(params["text_to_multi_labels"]):
        with open(params["text_to_multi_labels"]) as f:
            text_to_multi_labels = json.load(f)
            rule_predictions = predict_by_rule(input_sentences, text_to_multi_labels)
    else:
        rule_predictions = [[None]] * len(input_sentences)

    # tf.logging.info("All rule predictions:")
    # tf.logging.info(len(rule_predictions))
    # pprint(rule_predictions)

    for file_count, file_size in enumerate(input_sentence_lengths):
        file_model_predictions = model_predictions[:file_size]
        file_rule_predictions = rule_predictions[:file_size]
        file_model_predictions = [p for p in file_model_predictions if p != "null:null:null:null:null"]
        file_rule_predictions = [p for pp in file_rule_predictions for p in pp]
        file_rule_predictions = [p for p in file_rule_predictions if p is not None]
        file_predictions = file_model_predictions + file_rule_predictions
        # print(file_predictions)

        if len(input_sentence_lengths) == 1:
            # pprint(file_predictions)
            tf.logging.info("Model prediction:")
            pprint(file_model_predictions)
            tf.logging.info("Rule prediction:")
            pprint(file_rule_predictions)
            tf.logging.info("All prediction:")
            pprint(file_predictions)
            tf.logging.info("File size: True: %s, Another: %s" % (len(file_rule_predictions), file_size))

        # write to xml
        xml_filename = os.path.basename(xml_files[file_count])
        xml_filepath = os.path.join(params["export_xml_path"], xml_filename)
        tf.logging.info("Writing %s: [%s/%s] .." % (xml_filename, file_count + 1, len(xml_files)))
        write2xml(file_predictions, xml_filepath)

        # delete elements
        del model_predictions[:file_size]
        del rule_predictions[:file_size]

        # if xml_filename == "110-01.xml":
        #     tf.logging.info("Model prediction:")
        #     pprint(file_model_predictions)
        #     tf.logging.info("Rule prediction:")
        #     pprint(file_rule_predictions)
        #     tf.logging.info("All prediction:")
        #     pprint(file_predictions)
        #     return



    # sentence_id = 1
    # file_labels = []
    # file_counter = 0
    # # print(input_sentences)
    # for pred, rule_pred in zip(pred_ids, rule_outputs):
    #     if len(input_sentence_lengths) <= 0:
    #         break
    #     # print(input_sentence_lengths, sentence_id)
    #     label = label_encoder.decode(pred)
    #     if sentence_id < input_sentence_lengths[0]:
    #         file_labels.append(label)
    #         #####
    #         if rule_pred is not None:
    #             file_labels.extend(rule_pred)
    #         sentence_id += 1
    #     else:
    #         del input_sentence_lengths[0]
    #         # write to file
    #         tf.logging.info("Writing [%s/%s] .." % (file_counter, len(xml_files)))
    #         xml_filename = os.path.basename(xml_files[file_counter])
    #         xml_filepath = os.path.join(params["export_xml_path"], xml_filename)
    #         file_labels.append(label)
    #         #####
    #         if rule_pred is not None:
    #             file_labels.extend(rule_pred)
    #         write2xml(file_labels, xml_filepath)
    #         # update file counter
    #         sentence_id = 1
    #         file_counter += 1
    #         file_labels = []
        # update start

    # for xml_file in xml_files:
    #     tf.logging.info("Processing %s ..." % xml_file)
    #     # predict input fn
    #     input_sentences = read_xml_text(xml_file)
    #
    #     input_examples = [bert.run_classifier.InputExample(
    #         guid="",
    #         text_a=x, text_b=None, label="null:null:null:null:null"
    #     ) for x in input_sentences]
    #     input_features = bert.run_classifier.convert_examples_to_features(
    #         input_examples, label_list,
    #         params["max_sequence_length"],
    #         tokenizer
    #     )
    #     predict_input_fn = bert.run_classifier.input_fn_builder(
    #         features=input_features,
    #         seq_length=params["max_sequence_length"],
    #         is_training=False,
    #         drop_remainder=False
    #     )
    #
    #     pred_ids = estimator.predict(input_fn=predict_input_fn)
    #     this_labels = []
    #     counter = 0
    #     for pred in pred_ids:
    #         label = label_encoder.decode(pred)
    #         this_labels.append(label)
    #         if os.path.isfile(xmlpath):
    #             print("Counter:", counter)
    #             print(input_sentences[counter])
    #             print(label)
    #             print("="*10)
    #             counter += 1
    #
    #     xml_filename = os.path.basename(xml_file)
    #     xml_filepath = os.path.join(params["export_xml_path"], xml_filename)
    #     write2xml(this_labels, xml_filepath)



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run(main)
