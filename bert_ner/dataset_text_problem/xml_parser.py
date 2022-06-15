import re
import os
import json
import collections
from lxml import etree
# from nltk.tokenize import sent_tokenize
from third_party.segmenter import split_single

default_status = "null"
tag_dict = {
    "risk_factor": default_status,
    "time": default_status,
    "indicator": default_status,
    "status": default_status,
    "type1": default_status
}

class XmlParser:
    tag_types = {"DIABETES": "time:indicator",
                 "CAD": "time:indicator",
                 "HYPERTENSION": "time:indicator",
                 "HYPERLIPIDEMIA": "time:indicator",
                 "SMOKER": "status",
                 "OBESE": "time:indicator",
                 "FAMILY_HIST": "indicator",
                 "MEDICATION": "time:type"}

    def __init__(self, label_separator=":"):
        self.label_separator = label_separator

    def read_xml(self, filepath):
        with open(filepath) as f:
            raw = f.read()
        root = etree.fromstring(raw.encode("utf-8"))
        return root

    def get_label(self, root):
        text_map_label = collections.defaultdict(str)
        text_map_multi_label = collections.defaultdict(list)
        for factor in self.tag_types.keys():
            for element in root.find("TAGS").findall(factor):
                for sub_element in element:
                    key_text = sub_element.attrib.get("text", "")
                    key_text = key_text.strip()

                    if not key_text:
                        continue
                    # remove extra whitespace
                    key_text = re.sub(r"\s{2,}", " ", key_text)

                    # label
                    label = self.get_label_from_element(sub_element, factor)
                    label_str = str.join(self.label_separator, label)

                    if label_str == "SMOKER:null:null:unknown:null":
                        label_str = "null:null:null:null:null"
                        continue

                    # multi-labels
                    if key_text in text_map_label:
                        if label_str != text_map_label[key_text]:
                            if label_str not in text_map_multi_label[key_text]:
                                text_map_multi_label[key_text].append(label_str)
                    else:
                        text_map_label[key_text] = label_str
                        text_map_multi_label[key_text] = [label_str]

        text_map_label = collections.defaultdict(str)
        for text, labels in text_map_multi_label.items():
            labels = sorted(labels)
            # text_map_label[text] = str.join("-", labels)
            text_map_label[text] = labels[0]
        # return text_map_label
        return text_map_label, text_map_multi_label

    def process_one(self, filepath, text_only=False):
        root = self.read_xml(filepath)
        text = root.find("TEXT").text
        text = text.replace("\n", " ")
        sent_list = split_single(text)
        sent_list = [re.sub(r"\s{2,}", " ", sent.strip()) for sent in sent_list if sent.strip()]
        if text_only:
            return sent_list
        #
        text_map_label, text_map_multi_label = self.get_label(root)

        # label all sentences
        labels = []
        for sent in sent_list:
            label = self.label_sent(sent, text_map_label)
            labels.append(label)

        for text, label in text_map_label.items():
            sent_list.append(text)
            labels.append(label)

        return sent_list, labels, text_map_multi_label

    def process_all(self, file_dir, output_dir, dataset="train"):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        print("Looking at %s" % file_dir)
        text_to_multi_label_final = collections.defaultdict(list)
        text_writer = open(os.path.join(output_dir, dataset+".text.txt"), "w")
        tag_writer = open(os.path.join(output_dir, dataset+".tag.txt"), "w")
        sent_count = 0
        total_text_map_multi_label = collections.defaultdict(list)
        for file in os.listdir(file_dir):
            # print("\tProcessing %s" % file)
            if file.endswith(".xml"):
                sent_list, label_list, text_map_multi_label = self.process_one(os.path.join(file_dir, file))

                # update mapping
                if dataset == "train":
                    for text, labels in text_map_multi_label.items():
                        if text in total_text_map_multi_label:
                            total_text_map_multi_label[text].extend(labels)
                        else:
                            total_text_map_multi_label[text] = labels

                # write tokens to file
                text_writer.write(str.join("\n", sent_list) + "\n")
                tag_writer.write(str.join("\n", label_list) + "\n")
                sent_count += len(sent_list)

        if dataset == "train":
            text_to_multi_label_final = collections.defaultdict(list)
            for k, v in total_text_map_multi_label.items():
                text_to_multi_label_final[k] = list(set(v))

            with open(os.path.join(output_dir, "text_to_multi_labels.json"), "w") as fw:
                json.dump(text_to_multi_label_final, fw, indent=2, ensure_ascii=False)

        text_writer.close()
        tag_writer.close()
        print("Xml extraction done.")
        print("\t%s sentences." % sent_count)

    @staticmethod
    def split_into_sent(text):
        sent_list = split_single(text.replace("\n", " "))
        # sent_list = [l.strip().replace("\n", " ") for l in sent_list if l.strip()]
        sent_list = [l.strip()for l in sent_list if l.strip()]
        sent_list = [re.sub(r"\s{2,}", " ", l) for l in sent_list]
        return sent_list

    @staticmethod
    def get_label_from_element(element, factor):
        time = default_status
        status = default_status
        indicator = default_status
        type1 = default_status
        if factor != "FAMILY_HIST" and factor != "SMOKER":
            time = element.attrib.get("time").replace(" ", "_")
        if factor != "SMOKER" and factor != "MEDICATION":
            indicator = element.attrib.get("indicator").replace(" ", "_")
        if factor == "SMOKER":
            status = element.attrib.get("status").replace(" ", "_")
        if factor == "MEDICATION":
            type1 = element.attrib.get("type1").replace(" ", "_")
        result = [factor, time, indicator, status, type1]
        # result = [l for l in result if l is not None]
        return result

    def label_sent(self, sent, text_to_label):
        # sort by length
        text_to_label = dict(sorted(text_to_label.items(), key=len, reverse=True))
        for k, v in text_to_label.items():
            # if k in sent:
            if re.findall(r"\b%s\b" % re.escape(k), sent):
                return v
        return str.join(self.label_separator, [default_status] * 5)


def extract_all():
    file_dir = os.path.expanduser("~/data/hsu/i2b2/EMR/Track2-RiskFactors/complete")
    train_file_dir = os.path.expanduser("~/data/hsu/i2b2/EMR/training-RiskFactors-Complete-Set-MAE")
    test_file_dir = os.path.expanduser("~/data/hsu/i2b2/EMR/testing-RiskFactors-Complete")
    output_dir = os.path.expanduser("~/data/hsu/i2b2/train_n_dev")
    parser = XmlParser()
    # data.convert_a_xml(filepath)
    parser.process_all(train_file_dir, dataset="train", output_dir=output_dir)
    parser.process_all(test_file_dir, dataset="test", output_dir=output_dir)


def extract_one():
    train_file_dir = os.path.expanduser("~/data/hsu/i2b2/EMR/training-RiskFactors-Complete-Set-MAE")
    test_file_dir = os.path.expanduser("~/data/hsu/i2b2/EMR/testing-RiskFactors-Complete")
    output_dir = os.path.expanduser("~/data/hsu/i2b2/train_n_dev")
    filename = "100-03.xml"
    filepath = os.path.join(train_file_dir, filename)
    parser = XmlParser()
    # data.convert_a_xml(filepath)
    sent_list, labels = parser.process_one(filepath)
    counter = 0
    for sent, label in zip(sent_list, labels):
        print(sent)
        print(label)
        print("=" * 30)
        if label != "null:null:null:null:null":
            counter += 1

    print(counter, len(sent_list))


if __name__ == '__main__':
    extract_all()

