import json
import re
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.cluster import AgglomerativeClustering
from sklearn.utils import resample


## Loading the dataset
f = open(r'C:/Users/luukz/OneDrive/Documenten/Master studie BA&QM/Computer Science/Paper/TVs-all-merged.json')
data = json.load(f)

## Create a dataframe and split the modelID from the data
temp = []
for i in range(len(list(data.values()))):
    if len(list(data.values())[i]) == 1:
        temp.append(list(data.values())[i][0])
    else:
        for k in range(len(list(data.values())[i])):
            temp.append(list(data.values())[i][k])

df = pd.DataFrame(temp)

## Stating feature values from selected features brand, screen resolution and screen refresh rate
all_tv_brands = ['affinity', 'avue', 'azend', 'coby', 'compaq', 'contex', 'craig', 'curtisyoung', 'dynex', 'elo',
                    'epson', 'gpx', 'haier', 'hannspree', 'hiteker', 'hisense', 'insignia', 'jvc', 'lg', 'magnavox',
                    'mitsubishi', 'naxa', 'nec', 'optoma', 'panasonic', 'philips', 'proscan', 'pyle', 'rca',
                    'samsung', 'sanyo', 'sansui', 'seiki', 'sharp', 'sceptre', 'sigmac', 'sony', 'sunbritetv',
                    'supersonic', 'tcl', 'toshiba', 'upstar', 'venturer', 'venturer', 'viewsonic', 'vizio',
                    'westinghouse']
screen_resolutions = ['720p', '1080p', '4K']
screen_refresh_rates = ['50/60hz', '60hz', '120hz', '240hz', '600hz']


################## DATA CLEANING #####################
def cleaning_data(data):
    modelid_uncleaned = (data['modelID'])
    title_uncleaned = (data['title'])
    shop = (data['shop'])

    # Change all upper case letters in Model ID to lowercase
    ModelIDs = []
    for i in range(len(modelid_uncleaned)):
        temp1 = re.sub(r'[^\w\s]', '', modelid_uncleaned[i])
        temp1 = temp1.lower()
        ModelIDs.append(temp1)

    # Standardizing the notation for key features
    for i in range(len(title_uncleaned)):
        # Remove any symbols or spaces
        title_uncleaned[i] = title_uncleaned[i].replace('-', '')
        title_uncleaned[i] = title_uncleaned[i].replace('/', '')
        title_uncleaned[i] = title_uncleaned[i].replace(':', '')
        title_uncleaned[i] = title_uncleaned[i].replace('–', '')
        title_uncleaned[i] = title_uncleaned[i].replace(';', '')
        title_uncleaned[i] = title_uncleaned[i].replace('+', '')
        title_uncleaned[i] = title_uncleaned[i].replace('(', '')
        title_uncleaned[i] = title_uncleaned[i].replace(')', '')
        title_uncleaned[i] = title_uncleaned[i].replace('[', '')
        title_uncleaned[i] = title_uncleaned[i].replace('.', " ")
        title_uncleaned[i] = title_uncleaned[i].replace(',', " ")
        title_uncleaned[i] = title_uncleaned[i].replace('  ', " ")

        # Standardize the notation for Inch
        title_uncleaned[i] = title_uncleaned[i].replace('"', 'inch')
        title_uncleaned[i] = title_uncleaned[i].replace('\"', 'inch')
        title_uncleaned[i] = title_uncleaned[i].replace("'", " ")
        title_uncleaned[i] = title_uncleaned[i].replace('inches', 'inch')
        title_uncleaned[i] = title_uncleaned[i].replace('-inch', 'inch')
        title_uncleaned[i] = title_uncleaned[i].replace(' inch', 'inch')

        # Standardize the notation for Hertz
        title_uncleaned[i] = title_uncleaned[i].replace(' hz', 'hz')
        title_uncleaned[i] = title_uncleaned[i].replace('hertz', 'hz')
        title_uncleaned[i] = title_uncleaned[i].replace('Hertz', 'hz')
        title_uncleaned[i] = title_uncleaned[i].replace('-hz', 'hz')
        title_uncleaned[i] = title_uncleaned[i].replace(' hertz', 'hz')

    titles = []
    for i in range(len(data)):
        temp2 = re.sub(r'[^\w\s]', '', title_uncleaned[i])
        temp2 = temp2.lower()
        temp2 = temp2.split()

        titles.append(temp2)

    KeyTitles = []
    seen = set()
    for sublist in titles:
        for title in sublist:
            if title not in seen:
                KeyTitles.append(title)
                seen.add(title)

    tv_brand = [next((j for j, brand in enumerate(all_tv_brands) if brand in title), 0) for title in titles]
    tv_resolution = [next((j for j, resolution in enumerate(screen_resolutions) if resolution in title), 0) for title in
                     titles]
    tv_refresh_rate = [next((j for j, rate in enumerate(screen_refresh_rates) if rate in title), 0) for title in titles]

# New dataframe with only selected features
    cleaned_data = pd.DataFrame({
        'modelID': ModelIDs,
        'title': titles,
        'shop': shop,
        'brand': tv_brand,
        'resolution': tv_resolution,
        'refresh rate': tv_refresh_rate
    })

    return cleaned_data, KeyTitles

print(cleaning_data(df))

#################### FUNCTIONS ########################
# Cosine distance
def cosine_distance(a, b):
    similar = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    distance = 1 - similar
    return distance


# Binary matrix
def make_binary_matrix(titles, Keytitles):
    return np.array([[1 if key in title else 0 for title in titles] for key in Keytitles])

# Hash function to determine the hash values
def hash_func(x, a, b, p):
    return (a*x + b)%p


# Minhashing
def minhash(data, n, p):
    permutation_vector_1 = np.random.permutation(n)
    permutation_vector_2 = np.random.permutation(n)
    rows, cols, sigrows = len(data), len(data[0]), n

    sigmatrix =  np.full((n, cols), sys.maxsize)
    for r in range(rows):
        hashvalue = []
        for k in range(n):
            hashvalue.append(hash_func(r, permutation_vector_1[k], permutation_vector_2[k], p))

        for c in range(cols):
            if data[r][c] == 0:
                continue
            for i in range(sigrows):
                if sigmatrix[i][c] > hashvalue[i]:
                    sigmatrix[i][c] = hashvalue[i]

    return sigmatrix


# Create the signature matrix and determine the hash bands
def signature_matrix(data, Keytitles, bands, rows):
    p = 5233
    n = bands * rows
    titles = data['title']
    input_matrix = make_binary_matrix(titles, Keytitles)
    sig_matrix = minhash(input_matrix, n, p)


    hash_bands_final = []
    for i in range(bands):
        band = sig_matrix[i * rows:(i + 1) * rows]
        band_hashes = [''.join(map(str, map(int, band[:, j]))) for j in range(band.shape[1])]
        hash_bands_final.append(band_hashes)
    return input_matrix, sig_matrix, hash_bands_final


# Candidate matrix
def candidate_matrix(band_list, bands):
    candidate = np.zeros((len(band_list[1]), len(band_list[1])))

    for i in range(len(band_list[0])):
        for j in range(i + 1, len(band_list[0])):
            if any(band_list[c][i] == band_list[c][j] for c in range(bands)):
                candidate[i, j] = candidate[j, i] = 1

    return candidate


# Dissimilarity matrix
def disMatrix(candidate, input_matrix, data):
    # Selecting the features to use for the dissimilarity
    shop = data['shop']
    brand = data['brand']
    resolution = data['resolution']
    refresh_rate = data['refresh rate']

    dis_matrix = np.ones((len(candidate), len(candidate))) * sys.maxsize

    for i in range(len(candidate)):
        for j in range(i + 1, len(candidate)):
            if (candidate[i, j] == 1 and
                    brand[i] == brand[j] and
                    shop[i] != shop[j] and
                    resolution[i] == resolution[j] and
                    refresh_rate[i] == refresh_rate[j]):
                distance = cosine_distance(input_matrix[:, i], input_matrix[:, j])
                dis_matrix[i, j] = dis_matrix[j, i] = distance

    dis_matrix[dis_matrix == 0] = np.inf
    return dis_matrix


# Obtaining the actual duplicate pairs from the data
def real_duplicates(data):
    tv_IDs = data['modelID']
    actual_duplicates = []
    for modelID in tv_IDs:
        duplicate = np.where(tv_IDs == modelID)[0]
        if len(duplicate) > 1:
            actual_duplicates.extend(combinations(duplicate, 2))
    actual_duplicates = list(set(actual_duplicates))

    return actual_duplicates


# Obtaining potential duplicate pairs via hierarchical clustering
def clustering(dis_matrix, threshold):
    clusters = AgglomerativeClustering(metric='precomputed',
                                       linkage='complete',
                                       distance_threshold=threshold,
                                       n_clusters=None)
    clusters.fit(dis_matrix)

    # Finding groups of similar items
    potential_duplicates = [group for cluster_id in range(clusters.n_clusters_)
                            for group in combinations(np.where(clusters.labels_ == cluster_id)[0], 2)
                            if len(group) > 1]
    return potential_duplicates


# Final performance function
def performance(data, bands, rows, threshold):
    # performing the data cleaning and separating the data from the keytitles
    clean_data = cleaning_data(data)
    data = clean_data[0]; KeyTitles = clean_data[1]


    # constructing the necessary matrices
    matrices = signature_matrix(data, KeyTitles, bands, rows)
    binary_matrix = matrices[0]; band_matrix = matrices[2]
    candidates = candidate_matrix(band_matrix, bands)
    dis_matrix = disMatrix(candidates, binary_matrix, data)

    # obtaining the real duplicates from the data
    actual_duplicates = real_duplicates(data)
    # determining the potential duplicates via clustering
    potential_duplicates = clustering(dis_matrix, threshold)

    # the number of real and potential duplicates
    n_actual_duplicates = len(actual_duplicates)
    n_potential_duplicates = len(potential_duplicates)

    # comparing the duplicate selection of our model with the real duplicates
    true_positives=[]; false_positives=[]
    for i in range(0, n_potential_duplicates):
        if potential_duplicates[i] in actual_duplicates:
            true_positives.append(potential_duplicates[i])
        else:
            false_positives.append(potential_duplicates[i])

    # number of true positives/false positives/false negatives
    n_true_positives = len(true_positives)
    n_false_positives = len(false_positives)
    n_false_negatives = len(actual_duplicates) - n_true_positives

    # obtaining the fraction of comparisons that were made out of the total number of possible comparions
    n_comps = np.count_nonzero(candidates) / 2
    n_comps_possible = len(data) * (len(data)-1) * 0.5
    frac_comps = n_comps / n_comps_possible

    # Pair Quality and Pair Completeness
    PQ = n_true_positives / n_comps
    PC = n_true_positives / n_actual_duplicates

    precision = n_true_positives / (n_true_positives+n_false_positives)
    recall = n_true_positives / (n_true_positives+n_false_negatives)

    F1_score = (2*precision*recall) / (precision+recall)
    F1_star = (2*PQ*PC) / (PQ+PC)

    return n_potential_duplicates, n_true_positives, PQ, PC, F1_score, F1_star, frac_comps


####################### BOOTSTRAP #####################################
def bootstrap_function(data, n_bootstraps, bands, rows, threshold):
    performance_measures = np.zeros((n_bootstraps, 7))

    for i in range(n_bootstraps):
        bootstrap_sample = resample(data[['modelID', 'title', 'shop']], n_samples=len(data), random_state=i)
        train_set = bootstrap_sample.drop_duplicates()
        test_set = data.loc[~data.index.isin(train_set.index)].reset_index(drop=True)

        performance_measures[i, :] = performance(test_set, bands, rows, threshold)


    print(f"Number of predicted duplicates: {np.mean(performance_measures[:, 0])}")
    print(f"True positives: {np.mean(performance_measures[:, 1])}")
    print(f"Pair quality: {np.mean(performance_measures[:, 2])}")
    print(f"Pair completeness: {np.mean(performance_measures[:, 3])}")
    print(f"F1 Score: {np.mean(performance_measures[:, 4])}")
    print(f"F1 Star: {np.mean(performance_measures[:, 5])}")
    print(f"Fraction of comparisons: {np.mean(performance_measures[:, 6])}")

    return tuple(performance_measures[:, j] for j in range(7))

# Initialize lists to store the results
results_bootstrap = []
frac_comps, F1_score, F1_star, PQ, PC = ([] for i in range(5))

# Define the different values for the parameters: (bands, rows)
bootstrap_parameters = [(800, 1), (400, 2), (266, 3), (200, 4), (160, 5), (133, 6)]

# Loop through each set of parameters and perform bootstrapping
for bands, rows in bootstrap_parameters:
    result = bootstrap_function(df, 5, bands, rows, 0.5)
    results_bootstrap.append(result)

# Calculate the mean for each metric from the results
for result in results_bootstrap:
    frac_comps.append(np.mean(result[6]))
    F1_score.append(np.mean(result[4]))
    F1_star.append(np.mean(result[5]))
    PC.append(np.mean(result[3]))
    PQ.append(np.mean(result[2]))


####################### PLOTS ##############################

# Plots for each performance measure
plt.plot(frac_comps, F1_score, linewidth=2, color='teal', marker='o')
plt.grid()
plt.title('F1 score', fontsize=16, color='navy')
plt.xlabel('Fractions of comparisons')
plt.ylabel('F1_score')
plt.show()

plt.plot(frac_comps, F1_star, linewidth=2, color='teal', marker='o')
plt.grid()
plt.title('F1 star', fontsize=16, color='navy')
plt.xlabel('Fractions of comparisons')
plt.ylabel('F1_star')
plt.show()


plt.plot(frac_comps, PC, linewidth=2, color='teal', marker='o')
plt.grid()
plt.title('Pair completeness', fontsize=16, color='navy')
plt.xlabel('Fractions of comparisons')
plt.ylabel('PC')
plt.show()

plt.plot(frac_comps, PQ, linewidth=2, color='teal', marker='o')
plt.grid()
plt.title('Pair quality', fontsize=16, color='navy')
plt.xlabel('Fractions of comparisons')
plt.ylabel('PQ')
plt.show()
