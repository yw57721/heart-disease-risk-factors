#!/usr/bin/env bash
set -e

pretrained_bert=/data/projects/test/uncased_L-12_H-768_A-12
model_dir=/data/projects/test/i2b2_data/bert_model
train_dir=/data/projects/test/i2b2_data/train
label_list=$train_dir/tag_vocab.txt
num_train_epochs=3
num_train_steps=50000

cd ..

python3 -m i2b2.bert_train \
--pretrained_bert=$pretrained_bert \
--label_list=$label_list \
--model_dir=$model_dir \
--train_dir=$train_dir \
--num_train_steps=$num_train_steps

test_xml_path=/data/projects/test/i2b2_data/testing-RiskFactors-Gold/
export_xml_path=/data/projects/test/i2b2_data/testing-RiskFactors-predicted/
test_xml_file=/data/projects/test/i2b2_data/testing-RiskFactors-Gold/110-011.xml

python3 -m i2b2.predict_predict \
--pretrained_bert=$pretrained_bert \
--label_list=$label_list \
--model_dir=$model_dir \
--train_dir=$train_dir \
--xml_file=$test_xml_file \
--xml_path=$test_xml_path \
--export_xml_path=$export_xml_path

python3 -m i2b2.metrics.evaluate cr $export_xml_path $test_xml_path
