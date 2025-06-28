"""
In this file, we will implement the clustering algorithm:

input to all clustering functions:
ensemble_data = [(random_number, "model name"), (random_number, "model name"), (random_number, "model name")]

output:
cluster_data = [ ((random_number, "model name"), ()), ((random_number, "model name"), ()), ((random_number, "model name"), ())]

where cluster_data[i][0] is the cluster center and cluster_data[i][1:] is the list of data points in the cluster

We will implement the following clustering algorithms:

"""
from typing import List, Tuple, Dict
import random
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from scipy.linalg import eigh
import os
import glob
import google.generativeai as genai
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

# Global embedding cache to avoid repeated API calls for the same model names
_embedding_cache: Dict[str, np.ndarray] = {}

# Global distance mode setting
DISTANCE_MODE = "embedding"  # Options: "embedding", "score", "hybrid"

def configure_gemini_api(api_key: str = ""):
    """
    Configure the Gemini API with the provided API key.
    If no API key is provided, it will try to use the GOOGLE_API_KEY environment variable.
    """
    if api_key == "":
        api_key = os.getenv('GOOGLE_API_KEY')
        if api_key is None:
            raise ValueError("Google API key not provided. Please set GOOGLE_API_KEY environment variable or pass api_key parameter.")
    
    print(f"Using API key: {api_key}")
    genai.configure(api_key=api_key)
    print("Gemini API configured successfully")

def get_embedding(text: str, model_name: str = "models/embedding-001") -> np.ndarray:
    """
    Get embedding for a text using Google's Gemini embedding model.
    
    Args:
        text: Text to embed
        model_name: Name of the embedding model to use
        
    Returns:
        numpy array containing the embedding vector
    """
    # Check cache first
    if text in _embedding_cache:
        return _embedding_cache[text]
    
    try:
        # Generate embedding using Gemini
        result = genai.embed_content(
            model=model_name,
            content=text,
            task_type="semantic_similarity"
        )
        
        embedding = np.array(result['embedding'])
        
        # Cache the result
        _embedding_cache[text] = embedding
        
        return embedding
        
    except Exception as e:
        print(f"Error getting embedding for '{text}': {e}")
        # Fallback: return a random embedding vector
        fallback_embedding = np.random.randn(768)  # Typical embedding dimension
        _embedding_cache[text] = fallback_embedding
        return fallback_embedding

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1, vec2: numpy arrays representing embedding vectors
        
    Returns:
        cosine similarity value between -1 and 1
    """
    # Handle zero vectors
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return np.dot(vec1, vec2) / (norm1 * norm2)

def create_trial_folder() -> str:
    """
    Create a unique trial folder for saving experiment results.
    
    Returns:
        Path to the created trial folder
    """
    # Find existing trial folders
    existing_trials = glob.glob("results/trial_*")
    existing_numbers = []
    
    for trial in existing_trials:
        try:
            # Extract number from trial_x format
            num = int(trial.split('_')[1])
            existing_numbers.append(num)
        except (IndexError, ValueError):
            continue
    
    # Find next available number
    next_num = 1
    if existing_numbers:
        next_num = max(existing_numbers) + 1
    
    # Create the folder
    trial_folder = f"results/trial_{next_num}"
    os.makedirs(trial_folder, exist_ok=True)
    
    print(f"Created trial folder: {trial_folder}")
    return trial_folder

def generate_test_data(N: int, value_range: Tuple[float, float] = (0, 100)) -> List[Tuple[float, str]]:
    """
    Generate N ensembles, each with a random number paired with a unique LLM name.
    
    Args:
        N: Number of ensembles (data points) to generate
        value_range: Range for random number generation (min, max)
    
    Returns: List[Tuple[float, str]] - [(random_number, "unique_model_name"), ...]
    """
    base_model_names = [
        "GPT-4", "Claude-3", "Gemini-Pro", "LLaMA-2", "PaLM-2", 
        "Vicuna", "Alpaca", "Mistral", "Falcon", "MPT",
        "CodeT5", "StarCoder", "InstructGPT", "ChatGLM", "Baichuan",
        "Qwen", "Yi", "DeepSeek", "Phi-3", "Llama-3",
        "T5", "BERT", "RoBERTa", "DeBERTa", "ELECTRA",
        "XLNet", "GPT-3", "PaLM", "Chinchilla", "Gopher"
    ]
    
    # Ensure we have enough unique names for N ensembles
    unique_names = []
    for i in range(N):
        if i < len(base_model_names):
            unique_names.append(f"{base_model_names[i]}-ensemble-{i+1}")
        else:
            # Generate additional unique names if N > base names
            base_idx = i % len(base_model_names)
            variant_num = (i // len(base_model_names)) + 2
            unique_names.append(f"{base_model_names[base_idx]}-v{variant_num}-ensemble-{i+1}")
    
    # Generate test data with unique names
    test_data = []
    for i in range(N):
        random_number = random.uniform(value_range[0], value_range[1])
        test_data.append((random_number, unique_names[i]))

    return test_data

def repulsive_sampling(ensemble_data: List[Tuple[float, str]], min_distance: float = 5.0) -> List[Tuple[float, str]]:
    """
    Repulsive sampling: Select cluster centers that are at least min_distance apart.
    
    Args:
        ensemble_data: List of (score, model_name) tuples
        min_distance: Minimum distance required between cluster centers
    
    Returns:
        List of cluster centers [(score, model_name), ...]
    """
    centers = []
    remaining_points = ensemble_data.copy()
    
    while len(remaining_points) > 0:
        # pick a random point from remaining_points
        center = random.choice(remaining_points)
        centers.append(center)
        
        # Remove the selected center from remaining points
        remaining_points.remove(center)
        
        # Remove all points within min_distance of the selected center
        remaining_points = [p for p in remaining_points 
                          if distance(p, center) >= min_distance]
    
    return centers

def random_sampling(ensemble_data: List[Tuple[float, str]], k: int) -> List[Tuple[float, str]]:
    """
    Random sampling: Select k random points from the ensemble data.
    """
    return random.sample(ensemble_data, k)

def nearest_sampling(ensemble_data: List[Tuple[float, str]], k: int) -> List[Tuple[float, str]]:
    """
    Nearest sampling: Select k points from the ensemble data that are closest to the first point.
    """
    return sorted(ensemble_data, key=lambda x: distance(x, ensemble_data[0]))[:k]

def dpp_sampling(ensemble_data: List[Tuple[float, str]], k: int = None, sigma: float = 10.0) -> List[Tuple[float, str]]:
    """
    Determinantal Point Process (DPP) sampling: Select diverse points using DPP.
    
    Args:
        ensemble_data: List of (score, model_name) tuples
        k: Number of points to sample (if None, uses variable-size DPP)
        sigma: Bandwidth parameter for the RBF kernel (higher = more diverse selection)
    
    Returns:
        List of selected points that are diverse according to the DPP
    """
    n = len(ensemble_data)
    if n == 0:
        return []
    
    if k is None:
        k = max(1, n // 4)  # Default to selecting about 1/4 of the points
    
    if k >= n:
        return ensemble_data.copy()
    
    # Create kernel matrix using RBF (Gaussian) kernel
    # K[i,j] = exp(-||x_i - x_j||^2 / (2 * sigma^2))
    # Convert scores to float to handle both string and numeric inputs
    scores = []
    for point in ensemble_data:
        try:
            score = float(point[0])
            scores.append(score)
        except (ValueError, TypeError):
            print(f"Warning: Could not convert score '{point[0]}' to number in DPP sampling")
            scores.append(0.0)
    
    scores = np.array(scores)
    K = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            dist_sq = (scores[i] - scores[j]) ** 2
            K[i, j] = np.exp(-dist_sq / (2 * sigma ** 2))
    
    # For k-DPP, we need to rescale the kernel matrix
    # so that the expected size is approximately k
    eigenvals, eigenvecs = eigh(K)
    eigenvals = np.real(eigenvals)
    eigenvals = np.maximum(eigenvals, 0)  # Ensure non-negative
    
    # Rescale eigenvalues to get expected size k
    if np.sum(eigenvals) > 0:
        scale_factor = k / np.sum(eigenvals / (1 + eigenvals))
        eigenvals = eigenvals * scale_factor
    
    # Sample from the DPP
    selected_indices = sample_dpp(eigenvals, eigenvecs, k)
    
    return [ensemble_data[i] for i in selected_indices]

def sample_dpp(eigenvals: np.ndarray, eigenvecs: np.ndarray, max_size: int) -> List[int]:
    """
    Sample from a DPP given eigenvalues and eigenvectors.
    
    Args:
        eigenvals: Eigenvalues of the kernel matrix
        eigenvecs: Eigenvectors of the kernel matrix  
        max_size: Maximum number of points to sample
        
    Returns:
        List of selected indices
    """
    n = len(eigenvals)
    
    # Step 1: Sample which eigenvalues to include
    # For each eigenvalue λ, include it with probability λ/(1+λ)
    selected_eigenvals = []
    selected_eigenvecs = []
    
    for i in range(n):
        lam = eigenvals[i]
        if lam > 0 and random.random() < lam / (1 + lam):
            selected_eigenvals.append(lam)
            selected_eigenvecs.append(eigenvecs[:, i])
    
    if len(selected_eigenvals) == 0:
        # Fallback: select a random point
        return [random.randint(0, n-1)]
    
    # Limit the number of selected eigenvalues/eigenvectors
    if len(selected_eigenvals) > max_size:
        indices = random.sample(range(len(selected_eigenvals)), max_size)
        selected_eigenvals = [selected_eigenvals[i] for i in indices]
        selected_eigenvecs = [selected_eigenvecs[i] for i in indices]
    
    # Step 2: Sample the actual points using the selected eigenvectors
    selected_indices = []
    V = np.column_stack(selected_eigenvecs) if selected_eigenvecs else np.array([]).reshape(n, 0)
    
    for _ in range(len(selected_eigenvecs)):
        if V.shape[1] == 0:
            break
            
        # Compute probabilities for each remaining item
        probabilities = np.sum(V ** 2, axis=1)
        
        # Normalize probabilities
        if np.sum(probabilities) == 0:
            break
        probabilities = probabilities / np.sum(probabilities)
        
        # Sample an item according to these probabilities
        selected_item = np.random.choice(n, p=probabilities)
        selected_indices.append(selected_item)
        
        # Update V by removing the component in the direction of the selected item
        if V.shape[1] > 1:
            v_selected = V[selected_item, :]
            v_selected = v_selected / np.linalg.norm(v_selected) if np.linalg.norm(v_selected) > 1e-10 else v_selected
            
            # Gram-Schmidt orthogonalization to remove selected direction
            V = V - np.outer(V @ v_selected, v_selected)
            
            # Remove zero columns
            norms = np.linalg.norm(V, axis=0)
            V = V[:, norms > 1e-10]
        else:
            break
    
    return list(set(selected_indices))  # Remove duplicates

def assign_to_clusters(ensemble_data: List[Tuple[float, str]], centers: List[Tuple[float, str]]) -> List[Tuple[Tuple[float, str], Tuple[Tuple[float, str], ...]]]:
    """
    Assign each data point to the nearest cluster center.
    
    Args:
        ensemble_data: All data points
        centers: Cluster centers
    
    Returns:
        List of (center, (assigned_points, ...)) tuples
    """
    clusters = [(center, ()) for center in centers]
    
    for point in ensemble_data:
        if point not in centers:  # Don't assign centers to themselves
            # Find nearest center
            nearest_center_idx = 0
            min_dist = distance(point, centers[0])
            
            for i, center in enumerate(centers):
                dist = distance(point, center)
                if dist < min_dist:
                    min_dist = dist
                    nearest_center_idx = i
            
            # Add point to nearest cluster
            center, existing_points = clusters[nearest_center_idx]
            clusters[nearest_center_idx] = (center, existing_points + (point,))
    
    return clusters

def distance(p1: Tuple[float, str], p2: Tuple[float, str]) -> float:
    """
    Distance between two points using cosine similarity of their model name embeddings.
    
    Args:
        p1, p2: Tuples of (score, model_name)
        
    Returns:
        Distance value where 0 means identical and 2 means completely opposite
    """
    try:
        # Get embeddings for the model names
        embedding1 = get_embedding(p1[1])  # p1[1] is the model name
        embedding2 = get_embedding(p2[1])  # p2[1] is the model name
        
        # Calculate cosine similarity
        similarity = cosine_similarity(embedding1, embedding2)
        
        # Convert similarity to distance: 1 - similarity gives us a distance
        # where 0 means identical (similarity = 1) and 2 means completely opposite (similarity = -1)
        distance_value = 1 - similarity
        
        return distance_value
        
    except Exception as e:
        print(f"Warning: Error calculating embedding distance between '{p1[1]}' and '{p2[1]}': {e}")
        
        # Fallback: use the original score-based distance
        try:
            score1 = float(p1[0])
            score2 = float(p2[0])
            return abs(score1 - score2) / 100.0  # Normalize to similar range as cosine distance
        except (ValueError, TypeError):
            print(f"Warning: Could not convert scores '{p1[0]}' or '{p2[0]}' to numbers")
            return 1.0  # Default moderate distance

def plot_clustering_results(ensemble_data: List[Tuple[float, str]], 
                          clusters: List[Tuple[Tuple[float, str], Tuple[Tuple[float, str], ...]]], 
                          min_distance: float, title: str, trial_folder: str):
    """
    Plot the before and after clustering visualization as bar graphs showing frequency.
    
    Args:
        ensemble_data: Original data points
        clusters: Clustered data with centers and assigned points
        min_distance: The minimum distance used for clustering
        title: Title for the clustering method
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Before clustering - frequency histogram with model color coding
    # Convert scores to float to handle both string and numeric inputs
    scores = []
    models = []
    for point in ensemble_data:
        try:
            score = float(point[0])
            scores.append(score)
            models.append(point[1])
        except (ValueError, TypeError):
            print(f"Warning: Could not convert score '{point[0]}' to number, skipping point")
            continue
    
    # Get unique model names and assign colors
    unique_models = list(set(models))
    model_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_models)))
    model_color_map = {model: color for model, color in zip(unique_models, model_colors)}
    
    # Group data by score and model
    score_model_counts = {}
    for score, model in zip(scores, models):
        if score not in score_model_counts:
            score_model_counts[score] = Counter()
        score_model_counts[score][model] += 1
    
    # Create stacked bar chart for pre-clustering data
    unique_scores = sorted(score_model_counts.keys())
    bottom = np.zeros(len(unique_scores))
    
    # Create bars for each model
    for model in unique_models:
        model_frequencies = [score_model_counts[score].get(model, 0) for score in unique_scores]
        ax1.bar(unique_scores, model_frequencies, width=1, alpha=0.7, 
               color=model_color_map[model], edgecolor='black', linewidth=0.5,
               bottom=bottom, label=model)
        bottom += model_frequencies
    
    ax1.set_xlabel('Score')
    ax1.set_ylabel('Frequency (Quantity)')
    ax1.set_title(f'Before Clustering\n{len(ensemble_data)} Total Points')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add legend for models (only if more than one model)
    if len(unique_models) > 1:
        ax1.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Set integer ticks on y-axis
    ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # Plot 2: After clustering - stacked frequency histogram by cluster
    colors = plt.cm.Set3(np.linspace(0, 1, len(clusters)))
    
    # Collect all scores for each cluster
    cluster_score_counts = []
    cluster_centers = []
    
    for i, (center, assigned_points) in enumerate(clusters):
        # Convert center score to float
        try:
            center_score = float(center[0])
            cluster_centers.append(center_score)
        except (ValueError, TypeError):
            cluster_centers.append(0.0)
        
        # Include center and assigned points, converting all to float
        all_cluster_scores = [cluster_centers[i]]  # Use the converted center score
        for point in assigned_points:
            try:
                point_score = float(point[0])
                all_cluster_scores.append(point_score)
            except (ValueError, TypeError):
                continue  # Skip invalid scores
                
        cluster_count = Counter(all_cluster_scores)
        cluster_score_counts.append(cluster_count)
    
    # Get all unique scores across all clusters
    all_cluster_scores = set()
    for cluster_count in cluster_score_counts:
        all_cluster_scores.update(cluster_count.keys())
    all_cluster_scores = sorted(all_cluster_scores)
    
    # Create stacked bar chart
    bottom = np.zeros(len(all_cluster_scores))
    
    for i, cluster_count in enumerate(cluster_score_counts):
        cluster_frequencies = [cluster_count.get(score, 0) for score in all_cluster_scores]
        
        bars = ax2.bar(all_cluster_scores, cluster_frequencies, width=1, 
                      bottom=bottom, alpha=0.7, color=colors[i], 
                      edgecolor='black', linewidth=0.5,
                      label=f'Cluster {i+1} (Center: {cluster_centers[i]:.0f})')
        
        # Add center markers on top of bars
        center_score = cluster_centers[i]
        if center_score in all_cluster_scores:
            center_idx = all_cluster_scores.index(center_score)
            center_height = bottom[center_idx] + cluster_frequencies[center_idx]
            ax2.scatter(center_score, center_height + 0.1, s=100, c='red', 
                       marker='*', edgecolor='black', linewidth=1, zorder=10)
        
        bottom += cluster_frequencies
    
    ax2.set_xlabel('Score')
    ax2.set_ylabel('Frequency (Quantity)')
    ax2.set_title(f'After {title}\n{len(clusters)} Clusters (min_distance={min_distance})')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set integer ticks on y-axis
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.tight_layout()
    
    # Save the plot instead of showing it
    filename = f"{title.lower().replace(' ', '_')}_clustering.png"
    save_path = os.path.join(trial_folder, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"Saved: {save_path}")
    
    # Print clustering summary
    print(f"\nClustering Summary:")
    print(f"Original data points: {len(ensemble_data)}")
    print(f"Number of clusters: {len(clusters)}")
    print(f"Minimum distance: {min_distance}")
    print(f"\nCluster details:")
    
    for i, (center, assigned_points) in enumerate(clusters):
        # Convert center score to float for formatting
        try:
            center_score = float(center[0])
            center_score_str = f"{center_score:.2f}"
        except (ValueError, TypeError):
            center_score_str = str(center[0])
            
        print(f"  Cluster {i+1}: Center = {center[1]} (score: {center_score_str})")
        print(f"    Assigned points: {len(assigned_points)}")
        for point in assigned_points:
            # Convert point score to float for formatting
            try:
                point_score = float(point[0])
                point_score_str = f"{point_score:.2f}"
            except (ValueError, TypeError):
                point_score_str = str(point[0])
            print(f"      - {point[1]} (score: {point_score_str})")
        print()

def plot_model_distributions(ensemble_data: List[Tuple[float, str]], trial_folder: str, x_max: float = 1000):
    """
    Create a 3-subplot figure showing raw random number distributions for each model.
    
    Args:
        ensemble_data: List of (score, model_name) tuples
        trial_folder: Folder to save the plot
        x_max: Maximum value for x-axis (minimum is always 0)
    """
    # Group data by model
    model_data = {}
    for score, model in ensemble_data:
        if model not in model_data:
            model_data[model] = []
        # Convert score to float to handle both string and numeric inputs
        try:
            numeric_score = float(score)
            model_data[model].append(numeric_score)
        except (ValueError, TypeError):
            print(f"Warning: Could not convert score '{score}' to number, skipping...")
            continue
    
    unique_models = list(model_data.keys())
    n_models = len(unique_models)
    
    if n_models == 0:
        print("No models found in data")
        return
    
    # Create subplots - arrange in a row if 3 or fewer models, otherwise use grid
    if n_models <= 3:
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 6))
    else:
        # For more than 3 models, use a grid layout
        cols = 3
        rows = (n_models + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    
    # Handle single model case
    if n_models == 1:
        axes = [axes]
    elif n_models > 3:
        axes = axes.flatten()
    
    # Color map for consistency
    colors = plt.cm.tab20(np.linspace(0, 1, n_models))
    
    for i, model in enumerate(unique_models):
        scores = model_data[model]
        
        # Create histogram
        ax = axes[i] if n_models > 1 else axes[0]
        
        # Count frequency of each score
        score_counts = Counter(scores)
        unique_scores = sorted(score_counts.keys())
        frequencies = [score_counts[score] for score in unique_scores]
        
        # Create bar plot
        ax.bar(unique_scores, frequencies, width=1, alpha=0.7, 
               color=colors[i], edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution: {model}\n({len(scores)} data points)')
        ax.grid(True, alpha=0.3, axis='y')
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        # Set consistent x-axis limits for all plots
        ax.set_xlim(0, x_max)
        
        # Add statistics text
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        ax.text(0.05, 0.95, f'Mean: {mean_score:.1f}\nStd: {std_score:.1f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide unused subplots if using grid layout
    if n_models > 3:
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)
    
    plt.suptitle(f'Raw Random Number Distributions by Model\nTotal: {len(ensemble_data)} data points across {n_models} models', 
                 fontsize=14, y=0.98)
    plt.tight_layout()
    
    # Save the plot
    filename = "model_distributions.png"
    save_path = os.path.join(trial_folder, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    import json 

    # Configure Gemini API (requires GOOGLE_API_KEY environment variable)
    try:
        configure_gemini_api()
    except ValueError as e:
        print(f"Warning: {e}")
        print("Using fallback distance calculation without embeddings")

    # Create unique trial folder for this experiment run
    trial_folder = create_trial_folder()

    test_data = json.load(open("test2.json"))
    print(test_data)
    
    # Generate test data with N ensembles
    N = 20
    #test_data = generate_test_data(N=N, value_range=(0, 100))
    
    # print("Generated Test Data:")
    # for i, (score, name) in enumerate(sorted(test_data, key=lambda x: x[0])):
    #     print(f"{i+1:2d}. {score:6.2f} - {name}")
    
    # Create model distribution plots first
    plot_model_distributions(test_data, trial_folder, x_max=100)
    
    # Perform repulsive sampling
    min_distance = 10.0
    centers = repulsive_sampling(test_data, min_distance=min_distance)
    
    # Assign points to clusters
    clusters = assign_to_clusters(test_data, centers)
    
    # Plot results
    plot_clustering_results(test_data, clusters, min_distance, "Repulsive Clustering", trial_folder)

    # Show DPP sampling
    dpp_centers = dpp_sampling(test_data, k=len(centers), sigma=15.0)
    dpp_clusters = assign_to_clusters(test_data, dpp_centers)
    plot_clustering_results(test_data, dpp_clusters, min_distance, "DPP Clustering", trial_folder)
    
    # Show random sampling
    random_centers = random_sampling(test_data, k=len(centers))
    random_clusters = assign_to_clusters(test_data, random_centers)
    plot_clustering_results(test_data, random_clusters, min_distance, "Random Clustering", trial_folder)

    # Show nearest sampling
    nearest_centers = nearest_sampling(test_data, k=len(centers))
    nearest_clusters = assign_to_clusters(test_data, nearest_centers)
    plot_clustering_results(test_data, nearest_clusters, min_distance, "Nearest Clustering", trial_folder)
    
    print(f"\nAll results saved to: {trial_folder}")