""""
Script for getting measures of the topic distributions present in the comments of each post.
"""
import numpy as np
import pandas as pd
from statistics import mode

verbose = True


def converter(in_str):
    return np.fromstring(in_str[1:-1], sep=" ")


# read in csv file with per-document topic distributions
doc_topic_df = pd.read_csv("data/topic_model/10_topics/doc_topic_distributions.csv",
                           index_col=0, converters={'topic_dist': converter})
# drop entries with 'dark' as id (seems to be error in data)
doc_topic_df = doc_topic_df[doc_topic_df['id'] != 'dark']
# drop duplicates
doc_topic_df.drop_duplicates(subset="id", inplace=True)

# just get those for the comments
doc_topic_df = doc_topic_df[doc_topic_df["type"] == "comment"]
# can drop 'type' column
doc_topic_df.drop(columns=['type'], inplace=True)

# read in csv files that have all info associated with comments and posts
ttc_comments = pd.read_csv("data/ttcafterloss/comments.csv", index_col=0)
infertility_comments = pd.read_csv("data/infertility/comments.csv", index_col=0)
comments_df = pd.concat([ttc_comments, infertility_comments])
# drop entries with 'dark' as id (seems to be error in data)
comments_df = comments_df[comments_df['id'] != 'dark']
# drop duplicates
comments_df.drop_duplicates(subset="id", inplace=True)
# process parent id so it matches post IDs (i.e., drop t1_ abbreviation)
comments_df["parent_post"] = comments_df.apply(lambda x: x["parent_id"].split("_")[1], axis=1)

ttc_posts = pd.read_csv("data/ttcafterloss/posts.csv", index_col=0)
infertility_posts = pd.read_csv("data/infertility/posts.csv", index_col=0)
post_df = pd.concat([ttc_posts, infertility_posts])

# merge comments df with doc topic df to get topics associated with each comment
comments_df = doc_topic_df.merge(comments_df, on="id")

# for each post, get its comments
mean_topic_dists = []
max_mean_topics = []
mode_max_topics = []
sample_max_topics = []

for idx, row in post_df.iterrows():
    post_id = row["id"]
    num_comments = row["num_comments"]
    # NOTE: here we are only getting comments that directly replied to the post
    # i.e., we are excluding comments that were made on other comments on the post
    comments = comments_df[comments_df["parent_post"] == post_id]
    if verbose:
        print("num comments is {} and # of comments found is {}".format(num_comments, len(comments)))
    mean_dist = np.NaN
    max_mean = np.NaN
    mode_max = np.NaN
    sample_max = np.NaN
    if not comments.empty:
        # get the topics for these comments
        topic_dist_matrix = np.stack(comments['topic_dist'].values)
        # get topic that each comment is most associated with
        comment_max_topics = np.argmax(topic_dist_matrix, axis=1)
        # get the topic that most posts are most associated with
        mode_max = mode(comment_max_topics)
        sample_max = np.random.choice(comment_max_topics)
        # get the mean of the topic distributions across comments
        mean_dist = np.mean(topic_dist_matrix, axis=0)
        # get the topic with the greatest mean value
        max_mean = np.argmax(mean_dist)
    mean_topic_dists.append(mean_dist)
    max_mean_topics.append(max_mean)
    mode_max_topics.append(mode_max)
    sample_max_topics.append(sample_max)

# save these with the rest of post data
post_df["mean_topic_dist"] = mean_topic_dists
post_df["max_mean_topic"] = max_mean_topics
post_df["mode_max_topic"] = mode_max_topics
post_df["max_topic_sample"] = sample_max_topics
post_df.to_csv("data/10_all_posts_with_comment_topics.csv")
