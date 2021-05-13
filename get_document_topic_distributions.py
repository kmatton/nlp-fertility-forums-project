"""
Script for applying topic model to corpus of documents to get the distribution of topics within those documents.
"""
import argparse
import ast
import os

import gensim
import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic_model_path", type=str, help="Path to topic model")
    parser.add_argument("--corpus_dir", type=str, help="Path directory containing corpus of documents to run model on.")
    parser.add_argument("--corpus_name", type=str, help="Prefix used in naming corpus files.")
    parser.add_argument("--output_dir", type=str, help="Path to directory to save computed topic dist features.")
    args = parser.parse_args()
    return args


def get_doc_topic_metrics(corpus_df, topic_model, output_dir):
    # get tf and convert from strings to literal form
    doc_tf_list = list(corpus_df["tf"])
    doc_tf_list = [ast.literal_eval(x) for x in doc_tf_list]
    # get topic distribution for each document
    doc_topic_list = topic_model[doc_tf_list]
    doc_topic_matrix = np.array(doc_topic_list)
    doc_topic_matrix = doc_topic_matrix[:, :, 1]
    # drop unnecessary columns
    corpus_df = corpus_df[["id", "subreddit", "type"]]
    corpus_df["topic_dist"] = list(doc_topic_matrix)
    # save
    corpus_df.to_csv(os.path.join(output_dir, "doc_topic_distributions.csv"))


def main():
    args = _parse_args()

    # load data associated with corpus of documents to get topics for
    corpus_df = pd.read_csv(os.path.join(args.corpus_dir, "{}_corpus.csv".format(args.corpus_name)), index_col=0)

    # load topic model
    topic_model = gensim.models.wrappers.ldamallet.LdaMallet.load(args.topic_model_path)

    # compute and save metrics
    get_doc_topic_metrics(corpus_df, topic_model, args.output_dir)


if __name__ == "__main__":
    main()
