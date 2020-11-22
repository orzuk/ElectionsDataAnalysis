# Functions for Loading and analysis of election data
# Some parts modified from a python notebook from Harel Kein
# Call libraries
import pandas as pd
import numpy as np
import os
import io
from matplotlib import pyplot as plt
import statsmodels.api as sm
from sklearn.decomposition import PCA

# Path to datafiles - change to your directory!
DATA_PATH = 'C:/Users/Or Zuk/Google Drive/HUJI/Teaching/Lab_52568/Data/Elections'

parties_dict_2019a ={'אמת' : "עבודה", 'ג' : "יהדות התורה", 'דעם'  : "רעם בלד", 'ום'  : "חדש תעל", 'טב'  : "איחוד מפלגות הימין",
 'ל'  : "ישראל ביתנו", 'מחל'  : "הליכוד", 'מרצ'  : "מרצ", 'פה'  : "כחול לבן", 'שס'  : "שס",  'כ'  : "כולנו",  'נ'  : "ימין חדש",  'ז'  : "זהות",  'נר'  : "גשר"}

parties_dict_2019b ={'אמת' : "עבודה גשר", 'ג' : "יהדות התורה", 'ודעם'  : "הרשימה המשותפת", 'טב'  : "ימינה", 'כף'  : "עוצמה יהודית",
 'ל'  : "ישראל ביתנו", 'מחל'  : "הליכוד", 'מרצ'  : "המחנה הדמוקרטי", 'פה'  : "כחול לבן", 'שס'  : "שס"}

parties_dict_2020 ={'אמת' : "עבודה גשר מרצ", 'ג' : "יהדות התורה", 'ודעם'  : "הרשימה המשותפת", 'טב'  : "ימינה", 'נץ'  : "עוצמה יהודית",
 'ל'  : "ישראל ביתנו", 'מחל'  : "הליכוד",  'פה'  : "כחול לבן", 'שס'  : "שס"}


big_parties_2019a = parties_dict_2019a.keys()
big_parties_2019b = parties_dict_2019b.keys()
big_parties_2020 = parties_dict_2020.keys()

# Functions
# Get number of votes of all parties above threshold
def parties_votes_total(df, thresh):
    par = df.sum().sort_values(ascending=False)
    return par[par > thresh]


# Get votes of all parties (normalized)
def parties_votes(df, thresh):
    par = df.sum().div(df.sum().sum()).sort_values(ascending=False)
    return par[par > thresh]


# Bar plot for all parties with votes above a threshold
def party_bar(df, thresh, city):
    width = 0.3
    votes = parties_votes(df, thresh)  # total votes for each party
    n = len(votes)  # number of parties
    names = votes.keys()

    rev_names = [name[::-1] for name in list(names)]
    fig, ax = plt.subplots()  # plt.subplots()

    city_votes = df.loc[city,names] / df.loc[city,names].sum()
    all_bar = ax.bar(np.arange(n), list(votes), width, color='b')
    city_bar = ax.bar(np.arange(n)+width, list(city_votes), width, color='r')

    ax.set_ylabel('Votes percent')
    ax.set_xlabel('Parties Names')
    ax.set_title('Votes percent per party')
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(rev_names)
    ax.legend((all_bar[0], city_bar[0]), ('Israel', city[::-1]))
    plt.show()

    return fig, ax


# Plot histogram of votes for a particular party across all cities
def one_party_hist(df, party, nbins):
    votes_per_city = df.sum(axis=1)
    party_share = df[party] / votes_per_city

    plt.hist(party_share, nbins)
    plt.xlabel('Num. Votes')
    plt.ylabel('Freq.')
    plt.title('Histogram of ' + party[::-1])
    plt.show()


# Show party votes vs. city size
def party_size_scatter(df, party):
    votes_per_city = df.sum(axis=1)
    party_share = df[party] / votes_per_city

    plt.scatter(votes_per_city, party_share)
    plt.xlabel('Total Votes')
    plt.ylabel('Party %')
    plt.title('Votes for ' + party[::-1])
    plt.show()

# Show party votes for two parties
def two_parties_scatter(df, party1, party2):
    votes_per_city = df.sum(axis=1)
    party_share1 = df[party1] / votes_per_city
    party_share2 = df[party2] / votes_per_city

    plt.scatter(party_share1, party_share2)  # Here draw circles with area proportional to city size
    plt.xlabel(party1[::-1])
    plt.ylabel(party2[::-1])
    plt.title('Scatter for two parties ' )
    plt.show()

# Shot heatmap of all parties
def heatmap_corr(corr_mat, names):
    rev_names = [name[::-1] for name in list(names)]
    fig, ax = plt.subplots()
    im = ax.imshow(corr_mat, cmap=plt.get_cmap('viridis'))
    n = corr_mat.shape[0]  # get number of variables
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(rev_names)
    ax.set_yticklabels(rev_names)
    ax.set_title("Parties pairwise correlations")

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)  # **cbar_kw)
    cbar.ax.set_ylabel('votes correlation', rotation=-90, va="bottom")


# Bar plot for all parties with votes above a threshold for 3 different bars
def turnout_bar(p, q, q2, thresh, labels):
    width = 0.2
    names = p[p>thresh].keys()
    rev_names = [name[::-1] for name in list(names)]
    fig, ax = plt.subplots()  # plt.subplots()

    p1 = p[p>thresh]
    q1 = q[p>thresh]
    q2 = q2[p>thresh]
    n1 = len(p1)
    orig_bar = ax.bar(np.arange(n1), list(p1), width, color='b')
    adj_bar = ax.bar(np.arange(n1)+width, list(q1), width, color='r')
    adj_bar2 = ax.bar(np.arange(n1)+2*width, list(q2), width, color='g')

    ax.set_ylabel('Votes percent')
    ax.set_xlabel('Parties Names')
    ax.set_title('Votes percent per party 2019 with/without turnout adjustment')
    ax.set_xticks(np.arange(n1))
    ax.set_xticklabels(rev_names)
    ax.legend((orig_bar[0], adj_bar[0], adj_bar2[0]), labels)
    plt.show()

    return fig, ax


# Read election results from csv file
def read_election_results(year, analysis, run_in_colab=False):

#    if run_in_colab:
#        df_raw = pd.read_csv('votes per ' + analysis + ' ' + year + '.csv',
#                             encoding='iso-8859-8', index_col='שם ישוב').sort_index()
#    else:
    df_raw = pd.read_csv(DATA_PATH + '/votes per ' + analysis + ' ' + year + '.csv',
                        encoding='iso-8859-8', index_col='שם ישוב').sort_index()
    if year == '2019b' or year == '2020':
        df = df_raw.drop('סמל ועדה', axis=1)  # new column added in Sep 2019
    else:  # 2019a
        df = df_raw
    if year == '2020' and analysis == "city":
        df = df.drop('Unnamed: 37', axis=1)
    if year == '2020' and analysis == "ballot":
        df = df.drop('Unnamed: 41', axis=1)

    df = df[df.index != 'מעטפות חיצוניות']
    if analysis == 'city':
        first_col = 5
    else:  # ballot
        first_col = 9
    df = df[df.columns[first_col:]]  # removing "metadata" columns

    return df, df_raw


# Simulate elections
def simulate_elections(N_tilde, v):
    return (np.random.binomial(N_tilde, v))


# Make corrections to the votes
# Correct for voting turnout in cities/ballots (from lab2)
def simple_turnout_correction(df, v):
    p = df.sum().div(df.sum().sum())  # votes without correction
    q_hat = df.div(v, axis='rows')
    q_hat = q_hat.sum().div(q_hat.sum().sum())  # Simple correction

    return p, q_hat

# Correct for voting turnout in cities
def regression_turnout_correction(df, turnout):
    p = df.sum().div(df.sum().sum())  # votes without correction
    bzb = df.sum(axis=1) / turnout # can also be read from outside
    model = sm.OLS(bzb, df).fit() # linear regression WITHOUT intercept
    q = p * model.params  # M.coef_ # alpha_inv
    q = q / q.sum()  # Normalize
    return p, q, model  # alpha_inv

# Change dataframe to include unique index for each ballot. From Harel Kain
def adapt_df(df, parties, include_no_vote=False, ballot_number_field_name=None):
    df['ballot_id'] = df['סמל ישוב'].astype(str) + '__' + df[ballot_number_field_name].astype(str)
    df_yeshuv = df.index  # new: keep yeshuv
    df = df.set_index('ballot_id')
    eligible_voters = df['בזב']
    total_voters = df['מצביעים']
    df = df[parties]
    df['ישוב'] = df_yeshuv  # new: keep yeshuv
    df = df.reindex(sorted(df.columns), axis=1)
    if include_no_vote:
        df['לא הצביע'] = eligible_voters - total_voters
    return df
