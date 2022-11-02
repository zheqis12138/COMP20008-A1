# This is the file you will need to edit in order to complete assignment 1
# You may create additional functions, but all code must be contained within this file


# Some starting imports are provided, these will be accessible by all functions.
# You may need to import additional items

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy.linalg import norm
import json
import re
import os
import nltk
from nltk.stem.porter import *
# nltk.download('punkt')
# nltk.download('stopwords')
from nltk.corpus import stopwords
from collections import defaultdict, OrderedDict
from sklearn.feature_extraction.text import TfidfTransformer
import string


datafilepath = "data/data.json"
articlespath = 'data/football'


def task1():
    with open(datafilepath, 'r') as f:
        Data = json.load(f)
    return sorted(Data["teams_codes"])


def task2():
    # Initialize a dictionary that store the team code, total goal scored and conceded
    with open(datafilepath, 'r') as f:
        Data = json.load(f)
        teams_code = sorted(Data["teams_codes"])
    score_dict = {team: {"team_code": "", "goals_scored_by_team": 0, "goals_scored_against_team": 0} for team in
                  teams_code}
    # Fill the dictionary with data in the json file
    for club_dict in Data["clubs"]:
        score_dict[club_dict["club_code"]]["team_code"] = club_dict["club_code"]
        score_dict[club_dict["club_code"]]["goals_scored_by_team"] = club_dict["goals_scored"]
        score_dict[club_dict["club_code"]]["goals_scored_against_team"] = club_dict["goals_conceded"]

    # Store the dictionary data into a csv file
    df = pd.DataFrame(score_dict.values())
    df.to_csv("task2.csv", index=False)
    return score_dict


def task3():
    files = sorted(os.listdir(articlespath))
    score_dict = {}
    # Open all the file within football
    for file in files:
        with open(articlespath + "/" + file, 'r') as f:
            content = f.read()
            # Use regular expression to find all scores in each article
            pattern = "[^\d]\d{1,2}\-\d{1,2}[^\d]"
            score_content_p = ' '.join(re.findall(pattern, content))
            pattern2 = '\d{1,2}\-\d{1,2}'
            score_list = re.findall(pattern2, score_content_p)
            # Compute the largest score in each article
            if len(score_list) == 0:
                score_dict[file] = 0
            else:
                total_score = []
                # Find sums of all scores in a single file
                for score in score_list:
                    total_score.append(int(score.split("-")[0]) + int(score.split("-")[1]))
                # Compute the largest score in an article
                score_dict[file] = max(total_score)
                # score_dict = {score_dict}

    # Save the result as a csv file
    df = pd.DataFrame({"filenames": list(score_dict.keys()), "total_goals": list(score_dict.values())})
    df.to_csv('task3.csv', index=False)
    return score_dict


def task4():
    score_dict = task3()
    # Find the data that is used to plot
    scores_list = list(score_dict.values())
    plt.boxplot(scores_list)
    # Label and title the figure
    plt.title("The Boxplot about Total Goals on Each Articles")
    plt.ylabel("Total Goals")
    plt.tight_layout()
    # Save the figure as task4.png
    plt.savefig("task4.png")
    return


def task5():
    # Find names of all participating team
    with open(datafilepath, 'r') as f:
        Data = json.load(f)
        team_name_list = sorted(Data["participating_clubs"])
    # Initialize a dictionary that contains all participating teams
    teams_dict = {team_name: 0 for team_name in team_name_list}

    # Load all files
    files = os.listdir(articlespath)
    teams_re = '|'.join(team_name_list)
    for file in files:
        with open(articlespath + "/" + file, 'r') as f:
            content = f.read()
            # Clean data by eliminating all punctuations
            name_mentioned_list = re.findall(teams_re, content)
            # Count how many times each team are mentioned
            for team in team_name_list:
                if team in name_mentioned_list:
                    teams_dict[team] += 1

    # Transfer the dictionary to a csv file
    df = pd.DataFrame({"club_name": list(teams_dict.keys()), "number_of_mentions": list(teams_dict.values())})
    df.to_csv("task5.csv", index=False)
    # Plot the bar chart
    plt.clf()
    plt.bar(list(teams_dict.keys()), list(teams_dict.values()))
    # Title and label the bar chart
    plt.title("Times that each club are mentioned")
    plt.xlabel("Club names")
    plt.ylabel("Times that are mentioned")
    plt.xticks(rotation="vertical")
    plt.tight_layout()
    plt.savefig("task5.png")
    return


def task6():
    # Find names of all participating team
    with open(datafilepath, 'r') as f:
        Data = json.load(f)
        team_name_list = sorted(Data["participating_clubs"])
    # Initialize a dictionary that contains all participating teams
    teams_dict = {team_name: 0 for team_name in team_name_list}
    # Initialize a dictionary for times of two teams that are mentioned together
    comentioned_dict = {team_name: 0 for team_name in team_name_list}
    for team in comentioned_dict.keys():
        comentioned_dict[team] = {team_name: 0 for team_name in team_name_list}

    # Load all files
    files = os.listdir(articlespath)
    teams_re = '|'.join(team_name_list)
    for file in files:
        with open(articlespath + "/" + file, 'r') as f:
            content = f.read()
            # Clean data by eliminating all punctuations
            name_mentioned_list = re.findall(teams_re, content)
            # Count how many times each team are mentioned
            for team in team_name_list:
                if team in name_mentioned_list:
                    teams_dict[team] += 1
                    # Check if another team is mentioned together with the current team
                    for another_team in team_name_list:
                        if another_team in name_mentioned_list:
                            comentioned_dict[team][another_team] += 1

    # Calculate the similarity for every two teams
    # Initialize a similarity dictionary dictionary
    sim_dict = {team_name: 0 for team_name in team_name_list}
    for team in sim_dict.keys():
        sim_dict[team] = {team_name: 0 for team_name in team_name_list}
    for team1 in team_name_list:
        for team2 in team_name_list:
            if team1 == team2:
                sim_dict[team1][team2] = 1
            elif (teams_dict[team1] + teams_dict[team2]) != 0:
                similarity = (2 * comentioned_dict[team1][team2]) / (teams_dict[team1] + teams_dict[team2])
                sim_dict[team1][team2] = similarity
            else:
                sim_dict[team1][team2] = 0

    # Transfer the dictionary into a pandas dataframe
    sim_df = pd.DataFrame(sim_dict.values())
    sim_df.index = team_name_list

    # Use heatmap tp visualize the data
    plt.clf()
    sns.heatmap(sim_df)
    plt.title("The similarity score for each pair of clubs ")
    plt.tight_layout()
    plt.savefig("task6.png")
    return sim_df


def task7():
    # Find names of all participating team
    with open(datafilepath, 'r') as f:
        Data = json.load(f)
        team_name_list = sorted(Data["participating_clubs"])
    # Initialize a dictionary that contains all participating teams
    teams_dict = {team_name: 0 for team_name in team_name_list}

    # Load all files
    files = os.listdir(articlespath)
    teams_re = '|'.join(team_name_list)
    for file in files:
        with open(articlespath + "/" + file, 'r') as f:
            content = f.read()
            # Clean data by eliminating all punctuations
            name_mentioned_list = re.findall(teams_re, content)
            # Count how many times each team are mentioned
            for team in team_name_list:
                if team in name_mentioned_list:
                    teams_dict[team] += 1

    # Find x and y values for scatter plot
    score_dict = task2()
    mention_list = [mention_time for mention_time in teams_dict.values()]
    score_list = [data_dict["goals_scored_by_team"] for data_dict in score_dict.values()]
    # Create the scatter plot
    plt.clf()
    plt.scatter(mention_list, score_list)
    plt.title("Goals scored by teams vs times of mentioned by team")
    plt.xlabel("Times of mentioned by the team")
    plt.ylabel("Goals scored by the team")
    plt.tight_layout()
    plt.savefig("task7.png")
    return


def task8(filename):
    with open(filename, 'r') as f:
        content = f.read()
        # Remove all non-alphabetic characters
        translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        content = content.translate(translator)
        content = ''.join(list(filter(lambda x: str.isalpha(x) or x in [' ', '\n', '\t'], content)))
        # Convert all spacing characters to whitespace
        content = re.sub('[\s]+', ' ', content)
        # Change all uppercase characters to lower case
        content = content.lower()
        # Tokenize the resulting string into words
        content = nltk.word_tokenize(content)
        # Remove all stopwords in nltk’s list of English stopwords from the resulting list
        stop_words = set(stopwords.words('english'))
        content = [w for w in content if w not in stop_words]
        # Remove all remaining words that are only a single character long
        content = [w for w in content if len(w) != 1]
    return content


def cos_similarity(dict1, dict2):
    '''
    Compute the cosine similarity for two given tf-idf dictionaries
    '''
    sum = 0
    for word in dict1.keys():
        if word in dict2.keys():
            sum += float(format(dict1[word] * dict2[word], '.16f'))
    return float(format(sum, '.16f'))


def task9():
    files = os.listdir(articlespath)
    # Create a dictionary that store every word that appears in files
    tf_dict = {}
    idf_dict = {}
    tf_idf_dict = {}
    df_dict = defaultdict(int)
    # Find df value for each word, store in df_dict
    i = 1
    for file in files:
        file_path = articlespath + "/" + file
        content = task8(file_path)
        content_no_duplication = list(OrderedDict.fromkeys(content))
        for word in content_no_duplication:
            df_dict[word] += 1

    # Find the tf value of each word in each file, store in tf_dict
    for file in files:
        file_path = articlespath + "/" + file
        content = task8(file_path)
        word_tally = defaultdict(int)
        for word in content:
            word_tally[word] += 1
        tf_dict[file] = word_tally
        i += 1

    # Find the idf value for each word, store in idf_dict
    for word in df_dict.keys():
        idf_dict[word] = float(format(np.log((1 + len(files)) / (1 + df_dict[word])) + 1, '.16f'))

    # Find tf-idf value for each word in each file, store in word_tfidf_dict
    for file_name in tf_dict.keys():
        tf_idf_list = []
        word_tfidf_dict = {}
        for word in tf_dict[file_name]:
            tf_idf = float(format(tf_dict[file_name][word] * idf_dict[word], '.16f'))
            tf_idf_list.append(tf_idf)
        # Normalise the result
        norm_value = norm(tf_idf_list)
        i = 0
        for word in tf_dict[file_name]:
            word_tfidf_dict[word] = float(format(tf_idf_list[i] / norm_value, '.16f'))
            i += 1
        tf_idf_dict[file_name] = word_tfidf_dict

    # Calculate the cosine similarity for every pair of files, store in cos_similarity_dict
    cos_similarity_dict = {}
    for i in range(len(files) - 1):
        for j in range(i + 1, len(files)):
            cos_similarity_dict[list(tf_idf_dict.keys())[i] + ' ' + list(tf_idf_dict.keys())[j]] = cos_similarity(
                list(tf_idf_dict.values())[i], list(tf_idf_dict.values())[j])
    sorted_tuple_list = sorted(list(cos_similarity_dict.items()), key=lambda x: x[1], reverse=True)
    # Get the top-10 similar pairs, store in a list of tuples
    top_10 = sorted_tuple_list[:10]

    # Output the result into a csv file
    dataframe_dict = {'article1': [info[0].split()[0] for info in top_10],
                      'article2': [info[0].split()[1] for info in top_10],
                      'similarity': [info[1] for info in top_10]}
    df = pd.DataFrame(dataframe_dict)
    df.to_csv('task9.csv', index=False)
    return
