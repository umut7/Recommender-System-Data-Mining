import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr

# Load data from CSV files
def load_data(main_file, new_file):
    main_data = pd.read_csv(main_file, header=None)
    new_users = pd.read_csv(new_file, header=None)
    
    main_data.replace('?', np.nan, inplace=True)
    new_users.replace('?', np.nan, inplace=True)
    
    main_data = main_data.apply(pd.to_numeric, errors='coerce')
    new_users = new_users.apply(pd.to_numeric, errors='coerce')
    
    return main_data.values, new_users.values

# Impute missing values using row means
def impute_missing_values(data):
    row_means = np.nanmean(data, axis=1)
    inds = np.where(np.isnan(data))
    data[inds] = np.take(row_means, inds[0])
    return data

# Calculate cosine similarity
def calculate_cosine_similarity(u, v):
    return 1 - cosine(u, v)

# Calculate Euclidean distance
def calculate_euclidean_distance(u, v):
    return 1 / (1 + euclidean(u, v))

# Calculate Pearson correlation coefficient
def calculate_pearson_correlation(u, v):
    mask = ~np.isnan(u) & ~np.isnan(v)
    if np.sum(mask) < 2:
        return 0
    return pearsonr(u[mask], v[mask])[0]

# Predict scores using k-NN


def predict_score(data, new_user, k, metric, is_item_based):
    scores = []
    for item_idx in range(data.shape[1]):
        similarities = []
        for user_idx in range(data.shape[0]):
            if is_item_based:
                similarity = metric(data[user_idx, :], new_user)
            else:
                similarity = metric(data[:, item_idx], new_user)
            rating = data[user_idx, item_idx]
            similarities.append((similarity, rating))
        similarities.sort(reverse=True)
        k_similarities = similarities[:k]
        k_weights = np.array([sim for sim, _ in k_similarities])
        k_ratings = np.array([rating for _, rating in k_similarities])
        if np.sum(k_weights) == 0:
            score = np.nan
        else:
            score = np.sum(k_weights * k_ratings) / np.sum(k_weights)
        scores.append(score)
    return scores

def calculate_cosine_similarity(u, v):
    # Ensure arrays have the same shape
    min_len = min(len(u), len(v))
    u = u[:min_len]
    v = v[:min_len]
    # Handle arrays with constant values
    if np.all(u == u[0]) or np.all(v == v[0]):
        return np.nan
    # Compute cosine similarity
    return 1 - cosine(u, v)

def calculate_pearson_similarity(u, v):
    # Ensure arrays have the same shape
    min_len = min(len(u), len(v))
    u = u[:min_len]
    v = v[:min_len]
    # Handle arrays with constant values
    if np.all(u == u[0]) or np.all(v == v[0]):
        return np.nan
    # Compute Pearson correlation coefficient
    mask = ~np.isnan(u) & ~np.isnan(v)
    if np.sum(mask) < 2:
        return np.nan
    return pearsonr(u[mask], v[mask])[0]

def calculate_euclidean_similarity(u, v):
    # Ensure arrays have the same shape
    min_len = min(len(u), len(v))
    u = u[:min_len]
    v = v[:min_len]
    # Compute Euclidean distance
    return euclidean(u, v)

def generate_report(main_file, new_users_file, output_file):
    main_data = np.genfromtxt(main_file, delimiter=',', skip_header=1)
    new_users_data = np.genfromtxt(new_users_file, delimiter=',', skip_header=1)
    
    row_means = np.nanmean(main_data, axis=1)
    imputed_main_data = np.where(np.isnan(main_data), np.expand_dims(row_means, axis=1), main_data)
    
    new_user_means = np.nanmean(new_users_data, axis=1)
    imputed_new_user = np.where(np.isnan(new_users_data), np.expand_dims(new_user_means, axis=1), new_users_data)
    
    k_values = [1, 2, 3, 4, 5]
    similarity_metrics = {
        "Cosine similarity": calculate_cosine_similarity,
        "Euclidean similarity": calculate_euclidean_similarity,
        "Pearson similarity": calculate_pearson_similarity
    }
    
    with open(output_file, 'w') as f:
        for metric_name, metric_func in similarity_metrics.items():
            is_item_based = True
            f.write(f"Item-based collaborative filtering recommendation scores for item {main_data.shape[1]}\n")
            f.write(f"{metric_name}\t")
            for k in k_values:
                f.write(f"k={k}\t")
            f.write("\n")
            
            for user_idx, new_user in enumerate(imputed_new_user):
                f.write(f"new-user{user_idx + 1}\t")
                scores = predict_score(imputed_main_data, new_user, k, metric_func, is_item_based)
                for score in scores:
                    if np.isnan(score):
                        f.write("N/A\t")
                    else:
                        f.write(f"{score:.2f}\t")
                f.write("\n")
        
        for metric_name, metric_func in similarity_metrics.items():
            is_item_based = False
            f.write(f"User-based collaborative filtering recommendation scores for item {main_data.shape[1]}\n")
            f.write(f"{metric_name}\t")
            for k in k_values:
                f.write(f"k={k}\t")
            f.write("\n")
            
            for user_idx, new_user in enumerate(imputed_new_user):
                f.write(f"new-user{user_idx + 1}\t")
                scores = predict_score(imputed_main_data, new_user, k, metric_func, is_item_based)
                for score in scores:
                    if np.isnan(score):
                        f.write("N/A\t")
                    else:
                        f.write(f"{score:.2f}\t")
                f.write("\n")

if __name__ == "__main__":
    generate_report('data-part1.csv', 'newusers-data-part1.csv', 'recommendation_report.txt')
