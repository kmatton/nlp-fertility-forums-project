"""
Preprocess Reddit post data
"""

import numpy as np
import pandas as pd

from text_utils import process_single_post_text


class DataProcessor:
    """
    Class to read and process Reddit post data.
    """
    def __init__(self, data_path, moderators_path):
        """
        :param data_path: path to CSV file with dataset
        :param moderators_path: path to text file with list of moderators

        This class contains functions to (1) filter posts according to the steps described below and
        (2) randomly assign posts to a train/test/val split.

        Current post filtering steps are:
        * remove posts made by moderators
        * remove posts that have been removed by moderators or deleted
        """
        self.data_df = pd.read_csv(data_path, index_col=0)
        self.moderators_path = moderators_path
        with open(self.moderators_path, 'r') as f:
            self.moderators = set(f.read().splitlines())

    def filter(self):
        keep_columns = [
            # id and other metadata
            'id',
            'created_utc',
            'author',
            'author_fullname',
            'author_flair_text',
            'url',
            # text
            'title',
            'selftext',
            # measures of post feedback (what we want to predict)
            'upvote_ratio',
            'score',
            'num_comments'
        ]
        self.data_df = self.data_df[keep_columns]

        # exclude posts from moderators
        self.data_df = self.data_df[~self.data_df["author"].isin(self.moderators)]

        # exclude deleted/removed posts
        del_list = ['[removed]', '[deleted]']
        self.data_df = self.data_df[~self.data_df["selftext"].isin(del_list)]

        # exclude posts without an author (this means they have been removed/deleted) and those without any text
        self.data_df = self.data_df[self.data_df["selftext"].notnull()]
        self.data_df = self.data_df[self.data_df["author"].notnull()]

    def assign_datasplit(self, train_frac=.6, val_frac=.2, test_frac=.2):
        assert train_frac + val_frac + test_frac == 1, "invalid data split fractions specified; must sum to 1"
        data_len = len(self.data_df)
        train_count = int(data_len * train_frac)
        val_count = int(data_len * val_frac)
        test_count = data_len - train_count - val_count
        assignments = ["train"] * train_count + ["val"] * val_count + ["test"] * test_count
        np.random.shuffle(assignments)
        self.data_df["data_split"] = assignments


class TextProcessor:
    """
    Class to read and process Reddit text
    """
    def __init__(self, data_path, lemmatize=True, remove_stops=True):
        """
        :param data_path: path to CSV file with dataset
        :param lemmatize: if True, apply lemmatization when pre-processing text
        :param remove_stops: if True, remove stopwords

        Current text pre-processing steps
        * Remove special characters and punctuation
        * Remove stopwords, except for those that are in the LIWC 2015 dictionary (optional)
        * Break into sentences and words
        * lemmatize (optional)

        The resulting dataframe has two new text columns, 'processed_text' and 'processed_title', which contain the
        pre-processed post body text and post title text respectively.
        The cleaned versions of the text are stored as lists of sentences, which are lists of words.

        NOTES:
        * We are currently not filtering based on word count, but may want to add this
        """
        self.data_df = pd.read_csv(data_path, index_col=0)
        self.lemmatize = lemmatize
        self.remove_stops = remove_stops

    def process_text(self):
        # process text for each title and post
        self.data_df['processed_text'] = []
        self.data_df['processed_title'] = []
        del_rows = []
        for idx, row in self.data_df.iterrows():
            row['processed_text'] = process_single_post_text(row['selftext'], do_lemmatize=self.lemmatize)
            row['processed_title'] = process_single_post_text(row['title'], do_lemmatize=self.lemmatize)
            # remove post if empty
            if not row['processed_text'] and not row['processed_title']:
                del_rows.append(idx)
        self.data_df.drop(index=del_rows, inplace=True)
