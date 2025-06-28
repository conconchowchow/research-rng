"""
Test data and examples for clustering.py
"""
import random
from clustering import repulsive_sampling, distance

def generate_test_data(n_points=20, value_range=(0, 100)):
    """
    Generate test data in the format expected by clustering functions.
    Returns: List[Tuple[float, str]] - [(random_number, "model name"), ...]
    """
    model_names = [
        "GPT-4", "Claude-3", "Gemini-Pro", "LLaMA-2", "PaLM-2", 
        "Vicuna", "Alpaca", "Mistral", "Falcon", "MPT",
        "CodeT5", "StarCoder", "InstructGPT", "ChatGLM", "Baichuan",
        "Qwen", "Yi", "DeepSeek", "Phi-3", "Llama-3"
    ]
    
    test_data = []
    for i in range(n_points):
        random_number = random.uniform(value_range[0], value_range[1])
        model_name = random.choice(model_names)
        # Add index to make model names unique
        unique_model_name = f"{model_name}-{i+1}"
        test_data.append((random_number, unique_model_name))
    
    return test_data

def generate_clustered_test_data():
    """
    Generate test data with natural clusters for better testing.
    """
    test_data = []
    
    # Cluster 1: High performance models (80-95 range)
    high_perf_models = ["GPT-4", "Claude-3", "Gemini-Pro"]
    for i, model in enumerate(high_perf_models):
        for j in range(3):  # 3 variants per model
            score = random.uniform(80, 95)
            test_data.append((score, f"{model}-variant-{j+1}"))
    
    # Cluster 2: Medium performance models (60-80 range)
    med_perf_models = ["LLaMA-2", "Mistral", "Falcon"]
    for i, model in enumerate(med_perf_models):
        for j in range(3):
            score = random.uniform(60, 80)
            test_data.append((score, f"{model}-variant-{j+1}"))
    
    # Cluster 3: Lower performance models (40-60 range)
    low_perf_models = ["Alpaca", "Vicuna", "MPT"]
    for i, model in enumerate(low_perf_models):
        for j in range(2):
            score = random.uniform(40, 60)
            test_data.append((score, f"{model}-variant-{j+1}"))
    
    # Add some outliers
    test_data.extend([
        (95.5, "Super-Model-X"),
        (35.2, "Basic-Model-A"),
        (25.8, "Experimental-Model-B")
    ])
    
    return test_data

def test_distance_function():
    """Test the distance function"""
    print("Testing distance function:")
    p1 = (10.0, "Model-A")
    p2 = (15.0, "Model-B")
    dist = distance(p1, p2)
    print(f"Distance between {p1} and {p2}: {dist}")
    
    p3 = (50.0, "Model-C")
    p4 = (30.0, "Model-D")
    dist2 = distance(p3, p4)
    print(f"Distance between {p3} and {p4}: {dist2}")

def test_repulsive_sampling():
    """Test the repulsive sampling function"""
    print("\nTesting repulsive sampling:")
    
    # Generate test data
    test_data = generate_clustered_test_data()
    print(f"Generated {len(test_data)} test points")
    
    # Sort by score for easier visualization
    test_data.sort(key=lambda x: x[0])
    print("\nTest data (sorted by score):")
    for i, (score, model) in enumerate(test_data):
        print(f"{i+1:2d}. {score:5.1f} - {model}")
    
    # Note: The current repulsive_sampling function has an issue - min_distance is not defined
    # We'll need to fix this in the clustering.py file
    
    print(f"\nNote: The repulsive_sampling function needs min_distance parameter to work properly.")

def main():
    """Run all tests"""
    print("=== Clustering Algorithm Test Data ===")
    
    # Test basic data generation
    print("1. Basic random test data:")
    basic_data = generate_test_data(10)
    for score, model in basic_data:
        print(f"  {score:.2f} - {model}")
    
    print("\n" + "="*50)
    
    # Test clustered data generation
    print("2. Clustered test data:")
    clustered_data = generate_clustered_test_data()
    clustered_data.sort(key=lambda x: x[0])
    for score, model in clustered_data:
        print(f"  {score:.2f} - {model}")
    
    print("\n" + "="*50)
    
    # Test distance function
    test_distance_function()
    
    print("\n" + "="*50)
    
    # Test repulsive sampling (with note about the issue)
    test_repulsive_sampling()

if __name__ == "__main__":
    main() 