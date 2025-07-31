#!/usr/bin/env python3
"""
Example usage of the Generator class for ensemble model responses.
"""

from generator import Generator

def main():
    # Example configuration with multiple models from different providers
    config = [
        {
            "model_name": "gpt-3.5-turbo", 
            "number_of_models": 2,
            "temperature": 0.7
        },
        {
            "model_name": "gpt-4", 
            "number_of_models": 1,
            "temperature": 0.5,
            "max_tokens": 1500
        },
        {
            "model_name": "claude-3-sonnet-20240229", 
            "number_of_models": 1,
            "temperature": 0.6
        },
        {
            "model_name": "claude-3-haiku-20240307", 
            "number_of_models": 1
        }
    ]
    
    # Initialize the generator
    generator = Generator(
        config=config,
        temperature=0.7,  # default temperature
        max_tokens=1000,  # default max tokens
        logs_dir="logs"   # directory for logs
    )
    
    # Print model information
    print("Generator Configuration:")
    model_info = generator.get_model_info()
    print(f"Total models: {model_info['total_models']}")
    print(f"Unique model names: {model_info['unique_model_names']}")
    print(f"Base models: {model_info['base_models']}")
    print()
    
    # Example prompts
    prompts = [
        "Explain quantum computing in simple terms",
        "Write a short story about a robot learning to paint",
        "What are the benefits and risks of artificial intelligence?"
    ]
    
    # Generate ensemble responses for each prompt
    for i, prompt in enumerate(prompts, 1):
        print(f"=== Prompt {i}: {prompt} ===")
        
        try:
            responses = generator.generate_response(prompt)
            
            print(f"Generated {len(responses)} responses:")
            for response, model_name in responses:
                print(f"\n--- {model_name} ---")
                # Truncate long responses for display
                display_response = response[:300] + "..." if len(response) > 300 else response
                print(display_response)
                
        except Exception as e:
            print(f"Error generating responses: {e}")
        
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main() 