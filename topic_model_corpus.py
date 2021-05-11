"""
Create Corpus of Documents and Vocabulary from Reddit data.
Used as input for topic model training and inference.
"""
import os
import time

import gensim.corpora as corpora
import numpy as np
import pandas as pd

from text_utils import build_bigram_model, make_bigrams_docs, lemmatize_docs, process_single_post_text


class Corpus:
    """
    Class to read Reddit post/comment data and form corpus of document text, document ids, and vocabulary.
    """
    def __init__(self, csv_files, downsample=True):
        """
        :param csv_files: list of CSV files with post/comment data
        :param downsample: if True, downsample so that an equal number of entries are used from each file
                           (e.g., equal number for ttcafterloss vs infertility and comments vs posts)
        """
        subreddits = ["ttcafterloss", "infertility"]

        print("starting to read data")
        init_time = time.time()

        # read in data
        dfs = []
        min_len = float('inf')
        for data_path in csv_files:
            df = pd.read_csv(data_path, index_col=0)
            # check if post
            if 'selftext' in df.columns:
                df['text'] = df.apply(lambda x: x['title'] + ' ' + x['selftext'], axis=1)
                df['type'] = "post"
            else:  # should be comment
                assert 'body' in df.columns, "found df without selftext or body column"
                df['text'] = df['body']
                df['type'] = "comment"
            df['subreddit'] = subreddits[0] if subreddits[0] in data_path else subreddits[1]
            # keep only needed columns
            df = df[['id', 'subreddit', 'type', 'text']]
            if len(df) < min_len:
                min_len = len(df)
            dfs.append(df)

        # if downsample, take 'min_len' examples from each df
        dfs_sampled = []
        if downsample:
            print("minimum length df was {}. downsampling others to match.".format(min_len))
            for df in dfs:
                idx = np.random.choice(np.arange(len(df)), size=min_len, replace=False)
                dfs_sampled.append(df.iloc[idx])

        final_dfs = dfs if not downsample else dfs_sampled
        self.data_df = pd.concat(final_dfs)
        print("finished reading data in time {}".format(time.time() - init_time))
        self.vocab_dict = None

    def make_corpus(self, vocab_path=None):
        """
        :param vocab_path: path to file with vocab to use. If None, will create a vocab from the preprocessed post text
        """
        # (1) pre-process text
        self.process_text()
        # (2) remove empty posts
        self.data_df = self.data_df[self.data_df["text"].notnull()]
        # (3) create vocab or load existing one
        if vocab_path:
            self.vocab_dict = corpora.dictionary.Dictionary.load(vocab_path)
        else:
            self.vocab_dict = self.create_vocab_dict()
        # (4) create tf representation of text
        self.data_df["tf"] = self.data_df.apply(lambda x: self.vocab_dict.doc2bow(x["text"]), axis=1)

    def save_corpus(self, output_dir, corpus_name):
        """
        :param output_dir: path to directory to save corpus + vocab to
        :param corpus_name: name of corpus to use in naming saved files
        """
        self.data_df.to_csv(os.path.join(output_dir, "{}_corpus.csv".format(corpus_name)))
        self.vocab_dict.save(os.path.join(output_dir, "{}_vocab.dct".format(corpus_name)))

    def process_text(self):
        print("starting to process text")
        init_time = time.time()
        text_list = []
        # clean and tokenize text for each post/comment
        for idx, row in self.data_df.iterrows():
            text_list.append(process_single_post_text(row['text'], do_lemmatize=False, remove_stops=True))
        # find and add bigrams
        # first collect all sentences
        sentence_list = [sent for doc in text_list for sent in doc]
        # train bigram model
        bigram_model = build_bigram_model(sentence_list)
        # identify bigram phrases within text and convert them to this
        text_list = make_bigrams_docs(text_list, bigram_model)
        # remove sentence boundaries --> just have each document as list of words
        text_list = [[word for sent in doc for word in sent] for doc in text_list]
        # lemmatize!
        text_list = lemmatize_docs(text_list)
        # store as updated text
        self.data_df['text'] = text_list
        print("finished processing text in time {}".format(time.time() - init_time))

    def create_vocab_dict(self, no_below=25, no_above=.5):
        """
        Create vocab dictionary (word id -> word) from all documents.
        :param no_below: remove words that have less than this many occurrences across all documents
        :param no_above: remove words that appear in more than this % of documents
        """
        vocab_dict = corpora.Dictionary(list(self.data_df["text"]))
        vocab_dict.filter_extremes(no_below=no_below, no_above=no_above)
        return vocab_dict
