## Setup
Install the following packages:
* gensim
* spacy
* nltk

Run ``python -m spacy download en_core_web_sm`` (needed for spacy).

To run the topic modeling code, you need to download the Mallet topic model from [here](http://mallet.cs.umass.edu/download.php).

## Data
Datasets for the r/ttcafterloss and r/infertility subreddits can be found in the data/<subreddit_name>/ directory.

The posts.csv files contain all posts between 1/1/2015 and 9/15/2020, with the following filtering steps applied:
* remove posts made by moderators
* remove posts that have been removed by moderators or deleted (including those without authors)
* remove posts without any 'selftext'/post text

In addition, these dataframes contain train/test/val split assignments in the "data_split" column.

The other columns are as follows:
* Post ID and other metadata: 'id', 'created_utc', 'author', 'author_fullname', 'author_flair_text', 'url'
* Text: 'title', 'selftext'
* Measures of post feedback (what we want to predict): 'upvote_ratio', 'score', 'num_comments'

Note that the post.csv files contain raw text; text pre-processing was **not** applied to them.

In addition, there are csv files named "processed_posts.csv". These files contain posts after the following text pre-processing steps have been applied:
* Removed special characters and punctuation
* Removed stopwords, except for those that are in the LIWC 2015 dictionary
* Broken up into sentences and words
* Lemmatized each word

The resulting csv files have two new text columns, 'processed_text' and 'processed_title', which contain the pre-processed post body text and post title text respectively.
The pre-processed versions of the text are stored as lists of sentences, which are lists of words.
