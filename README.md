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

Finally, there is a single csv file named "all_posts_with_comment_topics.csv" that has a row for all posts from the "posts.csv" files in both the data/infertility and data/ttcafterloss
directories. This file contains metrics related to the distribution over topics found in the comments that responded to each post. We used a topic model with
k=30 topics. This means for each comment, we have a 30-D vector that encodes the probability of each topic being present in
that comment. Therefore, for each post, we have a (# comments x 30) topic distribution matrix. We extract metrics from this matrix
that can be used as our prediction targets.
The columns with these metrics are:
* ``mean_topic_dist``: the mean distribution of topics across all comments (a 30-D vector with a value 0-1 for each topic)
* ``max_mean_topic``: the # (1-30) of the topic with the greatest mean value across all comments.
* ``mode_max_topic``: first, for each comment, we select the topic that had the max value. We then take the topic # (1-30) that was most frequently the 
max value topic across all comments.
* ``max_topic_sample``: first, for each comment, we select the topic that had the max value. We then randomly sample a topic # (1-30) from this
set of topics that had the max value for each comment.