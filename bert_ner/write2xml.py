import os
from pprint import pprint
from lxml import etree


def general_factor(root, factor, doc_id, time, indicator):
    doc_id = "DOC" + str(doc_id)
    factor = etree.SubElement(root, factor)
    factor.set("id", doc_id)
    factor.set("time", time)
    factor.set("indicator", indicator)
    return factor


def medication(root, doc_id, time, type1):
    factor = etree.SubElement(root, "MEDICATION")
    doc_id = "DOC" + str(doc_id)
    factor.set("id", doc_id)
    factor.set("time", time)
    factor.set("type1", type1)
    factor.set("type2", "")
    return factor


def family_hist(root, doc_id, indicator):
    doc_id = "DOC" + str(doc_id)
    factor = etree.SubElement(root, "FAMILY_HIST")
    factor.set("id", doc_id)
    factor.set("indicator", indicator)
    return factor


def smoke(root, doc_id, status):
    doc_id = "DOC" + str(doc_id)
    factor = etree.SubElement(root, "SMOKER")
    factor.set("id", doc_id)
    factor.set("status", status)
    return factor


def write2xml(label_list, export_xml_filename):
    root = etree.Element("root")
    tags = etree.SubElement(root, "TAGS")
    smoke_status = None
    family_hist_indicator = None
    doc_id = 0

    label_list = list(set(label_list))

    for label_ in label_list:
        for label in label_.split("-"):
            factor, time, indicator, status, type1 = label.split(":")
            if factor == "null":
                continue

            time = time.replace("_", " ")
            indicator = indicator.replace("_", " ")
            status = status.replace("_", " ")
            type1 = type1.replace("_", " ")

            if factor.lower() == "smoker":
                smoke(tags, doc_id, status)
                smoke_status = 1
            elif factor.lower() == "family_hist":
                family_hist(tags, doc_id, indicator)
                family_hist_indicator = 1
            elif factor.lower() == "medication":
                medication(tags, doc_id, time, type1)
            else:
                general_factor(tags, factor, doc_id, time, indicator)

            doc_id += 1

    if family_hist_indicator is None:
        family_hist(tags, doc_id, "not present")
        doc_id += 1

    if smoke_status is None:
        smoke(tags, doc_id, "unknown")

    et = etree.ElementTree(root)
    et.write(export_xml_filename, pretty_print=True)


if __name__ == '__main__':
    label_list = ["MEDICATION:after_DCT:null:null:beta_blocker"]
    write2xml(label_list, "test.xml")
