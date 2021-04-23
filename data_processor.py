"""
Preprocess Reddit post data
"""

import pandas as pd

from text_utils import process_single_post_text


class DataProcessor:
    """
    Class to read and process Reddit post data.
    """
    def __init__(self, data_path, lemmatize):
        """
        :param data_path: path to CSV file with dataset
        :param lemmatize: if True, apply lemmatization when pre-processing text
        TODO:
        * Filter moderators (need to get list of known moderators) - also automoderator if that's a thing
        * Filter out posts that have been removed?
        * Get more reliable karma scores and comment counts (use Reddit API and if not count scraped comments)
        * Add filtering of which posts to include --> need some minimum word count?
        * Add train/test val/split (60-20-20)
        * Copy LIWC dictionary and add path to it

        Current text pre-processing steps
        * Remove special characters and punctuation
        * Remove stopwords, except for those that are in the LIWC 2015 dictionary
        * Break into sentences and words
        * lemmatize (optional)


        The resulting dataframe has two new text columns, 'processed_text' and 'processed_title', which contain the
        pre-processed post body text and post title text respectively.
        The cleaned versions of the text are stored as lists of sentences, which are lists of words.
        """
        data_df = pd.read_csv(data_path)
        keep_columns = [
            'id',
            'author',
            'created_utc',
            'title',
            'selftext',
            'score',
            'num_comments'
        ]
        self.lemmatize = lemmatize
        self.data_df = data_df[keep_columns]
        self.process_text()

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
