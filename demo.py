import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import warnings
warnings.filterwarnings('ignore')

class RepulsiveClusteringDemo:
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        
    def generate_sample_data(self, n_samples=300, n_features=2, n_clusters=4):
        """Generate sample data with clusters of varying densities"""
        # Create main clusters
        X1, _ = make_blobs(n_samples=n_samples//2, centers=n_clusters//2, 
                          n_features=n_features, cluster_std=1.0, 
                          center_box=(-5, 5), random_state=self.random_state)
        
        # Create denser clusters
        X2, _ = make_blobs(n_samples=n_samples//2, centers=n_clusters//2, 
                          n_features=n_features, cluster_std=0.5, 
                          center_box=(2, 8), random_state=self.random_state+1)
        
        return np.vstack([X1, X2])
    
    def random_sampling(self, data, k):
        """Standard random sampling for cluster centers"""
        indices = np.random.choice(len(data), k, replace=False)
        return data[indices], indices
    
    def repulsive_sampling(self, data, k, min_distance=2.0):
        """Repulsive sampling - points must be min_distance apart"""
        n_points = len(data)
        selected_indices = []
        selected_points = []
        
        # Start with a random point
        first_idx = np.random.randint(n_points)
        selected_indices.append(first_idx)
        selected_points.append(data[first_idx])
        
        # Add points that are far enough from existing ones
        max_attempts = 1000
        while len(selected_points) < k and max_attempts > 0:
            candidate_idx = np.random.randint(n_points)
            candidate_point = data[candidate_idx]
            
            # Check if candidate is far enough from all selected points
            distances = [np.linalg.norm(candidate_point - sp) for sp in selected_points]
            
            if all(d >= min_distance for d in distances):
                selected_indices.append(candidate_idx)
                selected_points.append(candidate_point)
            
            max_attempts -= 1
            
        return np.array(selected_points), selected_indices
    
    def build_dpp_kernel(self, data, quality_weights=None, length_scale=1.0):
        """Build kernel matrix for Determinantal Point Process"""
        n = len(data)
        
        if quality_weights is None:
            quality_weights = np.ones(n)
        
        # Compute pairwise distances
        distances = squareform(pdist(data))
        
        # RBF similarity kernel
        similarity = np.exp(-distances**2 / (2 * length_scale**2))
        
        # Combine quality and similarity
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = np.sqrt(quality_weights[i] * quality_weights[j]) * similarity[i, j]
        
        return K
    
    def sample_dpp(self, K, k):
        """Sample k points from a Determinantal Point Process"""
        n = K.shape[0]
        
        # Eigendecomposition
        eigenvals, eigenvecs = eigh(K)
        eigenvals = np.maximum(eigenvals, 0)  # Ensure non-negative
        
        # Sample eigenvalues
        selected_eigenvals = []
        for i in range(len(eigenvals)):
            if len(selected_eigenvals) >= k:
                break
            # Probability of including eigenvalue i
            prob = eigenvals[i] / (1 + eigenvals[i])
            if np.random.random() < prob:
                selected_eigenvals.append(i)
        
        if not selected_eigenvals:
            # Fallback to random sampling
            return self.random_sampling(np.arange(n), min(k, n))
        
        # Sample from the selected eigenspace
        selected_eigenvecs = eigenvecs[:, selected_eigenvals]
        
        # Use a greedy approach to select diverse points
        selected_indices = []
        remaining_indices = list(range(n))
        
        for _ in range(min(k, len(selected_eigenvals))):
            if not remaining_indices:
                break
                
            best_idx = remaining_indices[0]
            best_score = -np.inf
            
            for idx in remaining_indices:
                # Score based on how well this point represents the eigenspace
                score = np.sum(selected_eigenvecs[idx, :]**2)
                
                # Penalize points close to already selected ones
                if selected_indices:
                    min_dist = min(np.linalg.norm(K[idx, :] - K[sidx, :]) 
                                 for sidx in selected_indices)
                    score *= min_dist
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        return selected_indices
    
    def dpp_sampling(self, data, k, length_scale=2.0):
        """DPP sampling for diverse point selection"""
        K = self.build_dpp_kernel(data, length_scale=length_scale)
        selected_indices = self.sample_dpp(K, k)
        return data[selected_indices], selected_indices
    
    def evaluate_diversity(self, points):
        """Evaluate diversity of selected points"""
        if len(points) < 2:
            return 0
        distances = pdist(points)
        return {
            'min_distance': np.min(distances),
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances)
        }
    
    def plot_comparison(self, data, k=6):
        """Create comprehensive visualization comparing all methods"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Generate samples using different methods
        random_centers, random_indices = self.random_sampling(data, k)
        repulsive_centers, repulsive_indices = self.repulsive_sampling(data, k, min_distance=2.0)
        dpp_centers, dpp_indices = self.dpp_sampling(data, k, length_scale=2.0)
        
        methods = [
            ('Random Sampling', random_centers, random_indices),
            ('Repulsive Sampling', repulsive_centers, repulsive_indices),
            ('DPP Sampling', dpp_centers, dpp_indices)
        ]
        
        # Plot sampling results
        for i, (method_name, centers, indices) in enumerate(methods):
            ax = axes[0, i]
            
            # Plot all data points
            ax.scatter(data[:, 0], data[:, 1], c='lightblue', alpha=0.6, s=30)
            
            # Plot selected centers
            ax.scatter(centers[:, 0], centers[:, 1], c='red', s=200, 
                      marker='x', linewidths=3, label='Selected Centers')
            
            # Draw circles around selected points to show coverage
            for center in centers:
                circle = plt.Circle(center, 1.5, fill=False, color='red', alpha=0.3)
                ax.add_patch(circle)
            
            ax.set_title(f'{method_name}\nSelected {len(centers)} centers')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot clustering results
        for i, (method_name, centers, indices) in enumerate(methods):
            ax = axes[1, i]
            
            # Perform k-means clustering with selected centers
            if len(centers) > 0:
                kmeans = KMeans(n_clusters=len(centers), init=centers, n_init=1)
                cluster_labels = kmeans.fit_predict(data)
                
                # Plot clusters
                scatter = ax.scatter(data[:, 0], data[:, 1], c=cluster_labels, 
                                   cmap='viridis', alpha=0.6, s=30)
                
                # Plot cluster centers
                ax.scatter(centers[:, 0], centers[:, 1], c='red', s=200, 
                          marker='x', linewidths=3)
                
                # Calculate and display diversity metrics
                diversity = self.evaluate_diversity(centers)
                ax.text(0.02, 0.98, 
                       f"Min dist: {diversity['min_distance']:.2f}\n"
                       f"Mean dist: {diversity['mean_distance']:.2f}", 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_title(f'{method_name} - Clustering Result')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print diversity statistics
        print("\nDiversity Statistics:")
        print("-" * 50)
        for method_name, centers, indices in methods:
            diversity = self.evaluate_diversity(centers)
            print(f"{method_name}:")
            print(f"  Min distance: {diversity['min_distance']:.3f}")
            print(f"  Mean distance: {diversity['mean_distance']:.3f}")
            print(f"  Std distance: {diversity['std_distance']:.3f}")
            print()
    
    def animate_repulsive_process(self, data, k=6, min_distance=2.0):
        """Visualize the step-by-step repulsive sampling process"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        selected_points = []
        selected_indices = []
        n_points = len(data)
        
        # Start with a random point
        first_idx = np.random.randint(n_points)
        selected_indices.append(first_idx)
        selected_points.append(data[first_idx])
        
        steps = [selected_points.copy()]
        
        # Add points iteratively
        max_attempts = 1000
        while len(selected_points) < k and max_attempts > 0:
            candidate_idx = np.random.randint(n_points)
            candidate_point = data[candidate_idx]
            
            distances = [np.linalg.norm(candidate_point - sp) for sp in selected_points]
            
            if all(d >= min_distance for d in distances):
                selected_indices.append(candidate_idx)
                selected_points.append(candidate_point)
                steps.append(selected_points.copy())
            
            max_attempts -= 1
        
        # Plot the process
        ax1.clear()
        ax1.scatter(data[:, 0], data[:, 1], c='lightblue', alpha=0.6, s=30)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(selected_points)))
        for i, (point, color) in enumerate(zip(selected_points, colors)):
            ax1.scatter(point[0], point[1], c=[color], s=200, marker='x', linewidths=3)
            circle = plt.Circle(point, min_distance, fill=False, color=color, alpha=0.5)
            ax1.add_patch(circle)
            ax1.text(point[0], point[1] + 0.3, f'{i+1}', ha='center', fontweight='bold')
        
        ax1.set_title(f'Repulsive Sampling Process\n{len(selected_points)} points selected')
        ax1.grid(True, alpha=0.3)
        
        # Show distance matrix
        if len(selected_points) > 1:
            dist_matrix = squareform(pdist(selected_points))
            im = ax2.imshow(dist_matrix, cmap='viridis')
            ax2.set_title('Distance Matrix Between Selected Points')
            ax2.set_xlabel('Point Index')
            ax2.set_ylabel('Point Index')
            plt.colorbar(im, ax=ax2)
            
            # Add text annotations
            for i in range(len(selected_points)):
                for j in range(len(selected_points)):
                    ax2.text(j, i, f'{dist_matrix[i,j]:.1f}', ha='center', va='center',
                            color='white' if dist_matrix[i,j] < dist_matrix.max()/2 else 'black')
        
        plt.tight_layout()
        plt.show()

def main():
    # Create demo instance
    demo = RepulsiveClusteringDemo(random_state=42)
    
    # Generate sample data
    data = demo.generate_sample_data(n_samples=200, n_clusters=4)
    
    print("Repulsive Sampling vs Regular Sampling Demo")
    print("=" * 50)
    
    # Show comprehensive comparison
    demo.plot_comparison(data, k=6)
    
    # Show step-by-step repulsive process
    print("\nStep-by-step Repulsive Sampling Process:")
    demo.animate_repulsive_process(data, k=6, min_distance=2.0)
    
    # Demonstrate effect of different parameters
    print("\nEffect of Different Minimum Distances:")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    min_distances = [1.0, 2.0, 3.0]
    for i, min_dist in enumerate(min_distances):
        centers, indices = demo.repulsive_sampling(data, k=8, min_distance=min_dist)
        diversity = demo.evaluate_diversity(centers)
        
        axes[i].scatter(data[:, 0], data[:, 1], c='lightblue', alpha=0.6, s=30)
        axes[i].scatter(centers[:, 0], centers[:, 1], c='red', s=200, 
                       marker='x', linewidths=3)
        
        for center in centers:
            circle = plt.Circle(center, min_dist, fill=False, color='red', alpha=0.3)
            axes[i].add_patch(circle)
        
        axes[i].set_title(f'Min Distance = {min_dist}\n'
                         f'Selected: {len(centers)} points\n'
                         f'Mean Distance: {diversity["mean_distance"]:.2f}')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()