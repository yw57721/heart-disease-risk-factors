
import os
import json
import pandas as pd
import tensorflow as tf
from absl import app
from absl import flags

from bert import run_classifier, optimization, tokenization, modeling
import bert


FLAGS = flags.FLAGS
flags.DEFINE_string("pretrained_bert", os.path.expanduser("~/downloads/uncased_L-12_H-768_A-12"),
                    "pretrained bert model directory.")
flags.DEFINE_string("label_list", os.path.expanduser("~/data/hsu/i2b2/train_n_dev/train/tag_vocab.txt"),
                    "label list directory.")
flags.DEFINE_integer("max_sequence_length", 64,
                     "maximum sequence length.")

flags.DEFINE_string("train_dir", os.path.expanduser("~/data/hsu/i2b2/train_n_dev/train"),
                    "Path of data.")
flags.DEFINE_string("model_dir", os.path.expanduser("~/data/hsu/i2b2/text_model"),
                    "Path of model.")

# flags.DEFINE_integer("num_train_epochs", 3,
#                      "Number of training epochs.")
flags.DEFINE_integer("num_train_steps", 150000,
                     "Number of training steps.")
flags.DEFINE_integer("batch_size", 32, "Number of batch size.")
# flags.DEFINE_float("learning_rate", 2e-5, "Initial learning rate.")
# flags.DEFINE_float("warmup_propoption", 0.1, "Initial learning rate.")
# flags.DEFINE_integer("save_checkpoints_steps", 500, "Number of batch size.")
# flags.DEFINE_integer("save_summary_steps", 100, "Number of batch size.")


def convert_to_features(params, dataset):
    tokenizer = bert.tokenization.FullTokenizer(vocab_file=os.path.join(params["pretrained_bert"], "vocab.txt"),
                                                do_lower_case=True)
    dataset = pd.read_csv(os.path.join(params["train_dir"], dataset+".csv"))
    dataset_examples = dataset.apply(lambda x: bert.run_classifier.InputExample(
        guid=None,
        text_a=x["text"],
        text_b=None,
        label=x["label"]
    ), axis=1)
    features = bert.run_classifier.convert_examples_to_features(dataset_examples, params["label_list"],
                                                                params["max_sequence_length"], tokenizer)
    return features


def count_sentences(filepath):
    dataset = pd.read_csv(filepath)
    return len(dataset)
# def train_input_fn(params):
#     train_features = convert_to_features(params, "train")
#     return bert.run_classifier.input_fn_builder(
#         features=train_features,
#         seq_length=params["max_sequence_length"],
#         is_training=True,
#         drop_remainder=False
#     )
#
#
# def eval_input_fn(params):
#     test_features = convert_to_features(params, "test")
#     return bert.run_classifier.input_fn_builder(
#         features=test_features,
#         seq_length=params["max_sequence_length"],
#         is_training=False,
#         drop_remainder=False
#     )

def model_fn(features, labels, mode, params):
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    loss, _, _, probs = bert.run_classifier.create_model(
        bert_config=params["bert_config"],
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        labels=label_ids,
        num_labels=params["num_labels"],
        use_one_hot_embeddings=False
    )

    ##########
    # init from pretrained model
    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    assignment_map, initialized_variable_names = modeling.get_assignment_map_from_checkpoint(
        tvars, params["init_checkpoint"]
    )
    tf.train.init_from_checkpoint(params["init_checkpoint"], assignment_map)

    tf.logging.info("*** Trainable Variables ***")
    for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
        tf.logging.info(" name = %s, shape = %s%s", var.name, var.shape, init_string)
    ##########

    if mode != tf.estimator.ModeKeys.PREDICT:
        train_op = bert.optimization.create_optimizer(
            loss=loss,
            init_lr=params["learning_rate"],
            num_train_steps=params["num_train_steps"],
            num_warmup_steps=params["num_warmup_steps"],
            use_tpu=False)

        if mode == tf.estimator.ModeKeys.EVAL:
            # pred_ids = tf.squeeze(tf.argmax(probs, axis=-1, output_type=tf.int32))
            pred_ids = tf.argmax(probs, axis=-1, output_type=tf.int32)
            accuracy = tf.metrics.accuracy(label_ids, pred_ids)
            recall = tf.metrics.recall(label_ids, pred_ids)
            # f1_score = tf.contrib.metrics.f1_score(label_ids, pred_ids)
            metrics = {
                "eval_loss": tf.metrics.mean(loss),
                "accuracy": accuracy,
                "recall": recall,
                # "f1_score": f1_score
            }
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
        else:
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    else:
        # pred_ids = tf.squeeze(tf.argmax(probs, axis=-1, output_type=tf.int32))
        pred_ids = tf.argmax(probs, axis=-1, output_type=tf.int32)
        return tf.estimator.EstimatorSpec(mode, predictions=pred_ids)


def main(_):
    batch_size = 32
    learning_rate = 2e-5
    num_train_epochs = 3
    warmup_propoption = 0.1
    save_checkpoints_steps = 500
    save_summary_steps = 100

    params = {}
    for k in FLAGS:
        params[k] = FLAGS[k].value

    params["save_checkpoints_steps"] = save_checkpoints_steps
    params["save_summary_steps"] = save_summary_steps
    params["learning_rate"] = learning_rate
    params["batch_size"] = batch_size
    params["init_checkpoint"] = os.path.join(params["pretrained_bert"], "bert_model.ckpt")

    num_examples = count_sentences(os.path.join(params["train_dir"], "train.csv"))

    if params["num_train_steps"] < 0:
        params["num_train_steps"] = int(num_examples / batch_size * num_train_epochs)

    params["num_warmup_steps"] = int(params["num_train_steps"] * warmup_propoption)

    with open(params["label_list"]) as f:
        label_list = f.read().split("\n")
    params["num_labels"] = len(label_list)
    params["label_list"] = label_list

    # bert config
    # with open(os.path.join(params["pretrained_bert"], "bert_config.json")) as f:
    #     params["bert_config"] = json.load(f)
    params["bert_config"] = modeling.BertConfig.from_json_file(os.path.join(params["pretrained_bert"], "bert_config.json"))

    run_config = tf.estimator.RunConfig(
        model_dir=params["model_dir"],
        save_summary_steps=params["save_summary_steps"],
        save_checkpoints_steps=params["save_checkpoints_steps"],
        keep_checkpoint_max=100
    )
    estimator = tf.estimator.Estimator(
        model_fn=model_fn, params=params, config=run_config)

    train_features = convert_to_features(params, "train")
    train_input_fn = bert.run_classifier.input_fn_builder(
            features=train_features,
            seq_length=params["max_sequence_length"],
            is_training=True,
            drop_remainder=False
        )
    test_features = convert_to_features(params, "test")
    eval_input_fn = bert.run_classifier.input_fn_builder(
        features=test_features,
        seq_length=params["max_sequence_length"],
        is_training=False,
        drop_remainder=False
    )

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=params["num_train_steps"])
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, throttle_secs=10)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    # estimator.train(input_fn=train_input_fn, max_steps=params["num_train_steps"])


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run(main)
