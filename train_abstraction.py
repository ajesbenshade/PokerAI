#!/usr/bin/env python3
"""
Learned Abstraction Training Script
Trains the learned abstraction model offline on 1M simulated hands
"""

import torch
import time
from config import Config
from equity_model import LearnedAbstractionTrainer

def main():
    """Train learned abstraction model"""
    print("Learned Abstraction Training")
    print("=" * 40)
    print(f"GPU Available: {torch.cuda.is_available()}")
    print(f"Training Hands: {Config.LEARNED_ABSTRACTION_TRAINING_HANDS}")
    print(f"Batch Size: {Config.LEARNED_ABSTRACTION_TRAIN_BATCH_SIZE}")
    print(f"Epochs: {Config.LEARNED_ABSTRACTION_TRAIN_EPOCHS}")
    print(f"Device: {Config.DEVICE}")
    
    # Initialize trainer
    trainer = LearnedAbstractionTrainer()
    
    # Train the model
    start_time = time.time()
    trained_model = trainer.train()
    training_time = time.time() - start_time
    
    print("\nTraining completed!")
    print(f"Total training time: {training_time:.2f}s")
    print(f"Hands per second: {Config.LEARNED_ABSTRACTION_TRAINING_HANDS / training_time:.1f}")
    
    # Test the trained model
    print("\nTesting trained model...")
    
    # Create test data
    test_input = torch.randn(10, 169 + 5, device=Config.DEVICE)  # Batch of 10 test states
    
    with torch.no_grad():
        buckets = trained_model(test_input)
        embeddings = trained_model.get_embedding(test_input)
        reconstructions = trained_model.reconstruct(test_input)
    
    print("Test Results:")
    print(f"  Bucket predictions shape: {buckets.shape}")
    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  Reconstruction shape: {reconstructions.shape}")
    print(f"  Bucket distribution: {torch.argmax(buckets, dim=-1).cpu().tolist()}")
    
    print("\nLearned abstraction model training completed successfully!")
    print("Model saved as 'learned_abstraction.pth'")

if __name__ == "__main__":
    main()
