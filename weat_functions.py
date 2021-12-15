import numpy as np
from os import path
from sklearn.decomposition import PCA
import random
from scipy.spatial.distance import cdist
import plotly.express as px
import copy
from scipy.stats import norm, pearsonr
import pandas as pd
import csv
import seaborn as sns
from matplotlib import pyplot as plt

#Functions
def meta_analysis(effect_sizes, variances):

    num_samples = len(effect_sizes)

    effect_size_arr_ES = np.array(effect_sizes)
    variances_arr_V = np.array(variances)

    variance_arr = np.array(variances)
    inverse_variance_arr_W = 1 / variance_arr

    total_var_q1 = np.sum(inverse_variance_arr_W * (effect_size_arr_ES ** 2))
    total_var_q2 = np.sum(inverse_variance_arr_W * (effect_size_arr_ES ** 2)) / np.sum(inverse_variance_arr_W)

    total_variance_Q = total_var_q1 - total_var_q2

    if total_variance_Q > (num_samples - 1):
        
        between_sample_c = np.sum(inverse_variance_arr_W) - np.sum(inverse_variance_arr_W ** 2) / np.sum(inverse_variance_arr_W)
        between_sample_var_sigma_sq = (total_variance_Q - (num_samples - 1)) / between_sample_c

    else:

        between_sample_var_sigma_sq = 0

    adjusted_variance_arr_V = variances_arr_V + between_sample_var_sigma_sq
    variance_weights_arr_v = 1 / adjusted_variance_arr_V

    combined_effect_size = np.sum(variance_weights_arr_v * effect_size_arr_ES) / np.sum(variance_weights_arr_v)
    variance_weights_inverse = 1 / np.sum(variance_weights_arr_v)

    standard_error = np.sqrt(variance_weights_inverse)
    hypothesis_test = combined_effect_size / standard_error

    p_value = norm.sf(hypothesis_test, loc = 0, scale = 1)
    
    return combined_effect_size, p_value

def meta_target_weat(Attribute_A, Target_W, embedding_dataframe, iterations):

    A_Words = [i for i in Attribute_A if i in embedding_dataframe.index]
    W_Words = [i for i in Target_W if i in embedding_dataframe.index]

    A_Vecs = embedding_dataframe.loc[A_Words].to_numpy()
    W_Vecs = embedding_dataframe.loc[W_Words].to_numpy()

    for i in range(iterations):
        rand = [random.randint(0,len(embedding_dataframe.index)-1) for _ in range(len(A_Words))]
        B_Vecs = embedding_dataframe.iloc[rand].to_numpy()

        A_sims = 1 - cdist(W_Vecs, A_Vecs, metric='cosine')
        B_sims = 1 - cdist(W_Vecs, B_Vecs, metric='cosine')

        A_means = np.mean(A_sims, axis = 1)
        B_means = np.mean(B_sims, axis = 1)

        combined_sims = np.concatenate((A_sims,B_sims),axis=1)
        std_devs = np.std(combined_sims,axis=1,ddof=1)
        vars = np.var(combined_sims,axis=1)

        weats = (A_means - B_means) / std_devs

        if i == 0:
            all_vars = copy.deepcopy(vars)
            all_weats = copy.deepcopy(weats)
        else:
            all_vars = np.column_stack((all_vars,vars))
            all_weats = np.column_stack((all_weats,weats))

    combined_effect_sizes = []

    for i in range(len(all_weats)):
        es = all_weats[i]
        var = all_vars[i]
        combined_es = meta_analysis(es,var)[0]
        combined_effect_sizes.append(combined_es)

    weat_dict = {W_Words[idx]:combined_effect_sizes[idx] for idx in range(len(combined_effect_sizes))}
    return weat_dict

def large_target_weat(Attribute_A, Attribute_B, Target_W, embedding_dataframe, feature_vector = False):

    A_Words = [i for i in Attribute_A if i in embedding_dataframe.index]
    B_Words = [i for i in Attribute_B if i in embedding_dataframe.index]
    W_Words = [i for i in Target_W if i in embedding_dataframe.index]

    random.shuffle(A_Words)
    random.shuffle(B_Words)

    max_len = min(len(A_Words),len(B_Words))

    A_Words = A_Words[:max_len]
    B_Words = B_Words[:max_len]

    A_Vecs = embedding_dataframe.loc[A_Words].to_numpy()
    B_Vecs = embedding_dataframe.loc[B_Words].to_numpy()
    W_Vecs = embedding_dataframe.loc[W_Words].to_numpy()

    A_sims = 1 - cdist(W_Vecs, A_Vecs, metric='cosine')
    B_sims = 1 - cdist(W_Vecs, B_Vecs, metric='cosine')

    A_means = np.mean(A_sims, axis = 1)
    B_means = np.mean(B_sims, axis = 1)

    combined_sims = np.concatenate((A_sims,B_sims),axis=1)
    std_devs = np.std(combined_sims,axis=1,ddof=1)

    weats = (A_means - B_means) / std_devs
    if feature_vector:
        return weats
    
    weat_dict = {W_Words[idx]:weats[idx] for idx in range(len(weats))}
    return weat_dict

def obtain_longitudinal_correlation(A,B,W,time_embeddings,ground_truth_series):

    target_group_weats = []
    
    for embedding in time_embeddings:
        
        embedding_df = pd.read_table(embedding,sep=' ',header=None,quoting=csv.QUOTE_NONE,index_col=0)
        weats = large_target_weat(A,B,W,embedding_df)
        mean_weats = np.mean(list(weats.values()))
        target_group_weats.append(mean_weats)
    
    return pearsonr(target_group_weats,ground_truth_series)

def meta_longitudinal_correlation(A,W,time_embeddings,ground_truth_series,iterations=1000):

    target_group_weats = []
    
    for embedding in time_embeddings:
        
        embedding_df = pd.read_table(embedding,sep=' ',header=None,quoting=csv.QUOTE_NONE,index_col=0)
        weats = meta_target_weat(A,W,embedding_df,iterations)        
        mean_weats = np.mean(list(weats.values()))
        target_group_weats.append(mean_weats)
    
    return pearsonr(target_group_weats,ground_truth_series)

def visualize_pca(vectors, labels, color_labels, num_components):

    if num_components == 2:

        pca = PCA(n_components = 2)
        components = pca.fit_transform(vectors, y = labels)
        variance = pca.explained_variance_ratio_.sum() * 100

        figure = px.scatter(
            components, x = 0, y = 1, color = color_labels,
            title = f'Total Explained Variance: {variance:.2f}%',
            hover_name = labels,
            labels = {'0': 'PC 1', '1': 'PC 2'}
            )

        figure.show()
        return

    if num_components == 3:

        pca = PCA(n_components = 3)
        components = pca.fit_transform(vectors, y = labels)
        variance = pca.explained_variance_ratio_.sum() * 100

        figure = px.scatter_3d(
            components, x = 0, y = 1, z = 2, color = color_labels,
            title = f'Total Explained Variance: {variance:.2f}%',
            hover_name = labels,
            text = labels,
            labels = {'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
            )

        figure.show()
        return

    else:

        pca = PCA(n_components = num_components)
        components = pca.fit_transform(vectors, y = labels)

        total_var = pca.explained_variance_ratio_.sum() * 100

        axis_labels = {str(i): f"PC {i+1}" for i in range(num_components)}

        fig = px.scatter_matrix(
            components,
            color = color_labels,
            dimensions = range(num_components),
            labels = axis_labels,
            hover_name = labels,
            title=f'Total Explained Variance: {total_var:.2f}%',
        )
        
        fig.update_traces(diagonal_visible=False)
        fig.show()
        return