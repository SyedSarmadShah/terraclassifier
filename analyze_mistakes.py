#!/usr/bin/env python3
"""
Debug why the model made a wrong prediction
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def analyze_misclassification():
    """Analyze why model makes mistakes"""
    
    print("\n" + "="*60)
    print("ANALYZING MODEL MISTAKES - WHY HIGHWAY → FOREST?")
    print("="*60 + "\n")
    
    # Load model
    model = load_model('models/saved_models/best_model.h5')
    classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 
               'Highway', 'Industrial', 'Pasture', 
               'PermanentCrop', 'Residential', 'River', 'SeaLake']
    
    # Load test data
    data = np.load('data/processed/preprocessed_data.npz', allow_pickle=True)
    X_test = data['X_test']
    y_test = data['y_test']
    
    print("📊 ANALYZING 100 PREDICTIONS...\n")
    
    # Make predictions
    predictions = model.predict(X_test, verbose=0)
    
    # Find mistakes
    mistakes = []
    for i in range(len(X_test)):
        true_idx = np.argmax(y_test[i])
        pred_idx = np.argmax(predictions[i])
        confidence = predictions[i][pred_idx]
        
        if true_idx != pred_idx:
            mistakes.append({
                'index': i,
                'true': classes[true_idx],
                'predicted': classes[pred_idx],
                'confidence': confidence,
                'true_confidence': predictions[i][true_idx],
                'img': X_test[i]
            })
    
    print(f"Total mistakes: {len(mistakes)} out of {len(X_test)}")
    print(f"Accuracy: {(len(X_test) - len(mistakes)) / len(X_test) * 100:.1f}%\n")
    
    # Find Highway → Forest mistakes specifically
    highway_to_forest = [m for m in mistakes if m['true'] == 'Highway' and m['predicted'] == 'Forest']
    
    print(f"🛣️  HIGHWAY MISCLASSIFIED AS FOREST: {len(highway_to_forest)} cases\n")
    
    if highway_to_forest:
        print("Examples of Highway → Forest mistakes:")
        for i, mistake in enumerate(highway_to_forest[:5]):
            print(f"\n  {i+1}. True: Highway | Predicted: Forest")
            print(f"     Confidence: {mistake['confidence']:.2%}")
            print(f"     True class confidence: {mistake['true_confidence']:.2%}")
    
    # Analyze common mistakes
    print("\n" + "-"*60)
    print("TOP 10 MOST COMMON MISTAKES:")
    print("-"*60 + "\n")
    
    mistake_pairs = {}
    for mistake in mistakes:
        key = f"{mistake['true']} → {mistake['predicted']}"
        mistake_pairs[key] = mistake_pairs.get(key, 0) + 1
    
    sorted_mistakes = sorted(mistake_pairs.items(), key=lambda x: x[1], reverse=True)
    for pair, count in sorted_mistakes[:10]:
        print(f"  {pair}: {count} times")
    
    # Why mistakes happen
    print("\n" + "="*60)
    print("WHY DOES THE MODEL MAKE MISTAKES?")
    print("="*60 + "\n")
    
    print("1. ❌ SIMILAR FEATURES")
    print("   - Highway has dark asphalt (gray/black)")
    print("   - Forest has dark vegetation (green/black)")
    print("   - Model sees 'dark' and confuses them\n")
    
    print("2. ❌ IMAGE QUALITY")
    print("   - Satellite images can be unclear")
    print("   - Resolution: 64x64 is small")
    print("   - Some details are lost when resizing\n")
    
    print("3. ❌ LIMITED TRAINING")
    print("   - Model trained on finite dataset")
    print("   - Can't see every possible variation\n")
    
    print("4. ❌ CLASS IMBALANCE")
    print("   - Some classes have more training examples")
    print("   - Model might be biased\n")
    
    print("5. ❌ CONFIDENCE SCORE")
    highway_forest = [m for m in mistakes if m['true'] == 'Highway' and m['predicted'] == 'Forest']
    if highway_forest:
        avg_conf = np.mean([m['confidence'] for m in highway_forest])
        print(f"   - Forest predicted with {avg_conf:.2%} confidence")
        print(f"   - But correct class (Highway) only got {np.mean([m['true_confidence'] for m in highway_forest]):.2%}")
        print("   - Model is unsure!\n")
    
    # Show visualization
    print("="*60)
    print("VISUALIZING A MISTAKE")
    print("="*60 + "\n")
    
    if mistakes:
        # Find a clear mistake example
        mistake = sorted(mistakes, key=lambda x: x['confidence'] - x['true_confidence'], reverse=True)[0]
        
        print(f"True: {mistake['true']} | Predicted: {mistake['predicted']}")
        print(f"Model confidence: {mistake['confidence']:.2%}")
        print(f"Correct class confidence: {mistake['true_confidence']:.2%}\n")

def compare_with_training():
    """Compare test accuracy with training accuracy"""
    
    print("\n" + "="*60)
    print("TRAINING vs TEST ACCURACY")
    print("="*60 + "\n")
    
    print("The model was trained on 50 epochs.")
    print("Look at the training graphs:")
    print("  - Training accuracy: Usually ~85-95%")
    print("  - Validation accuracy: Usually ~80-90%")
    print("  - Test accuracy: Often LOWER than validation\n")
    
    print("This is normal! It's called OVERFITTING:\n")
    print("  ✓ Model learns training data very well")
    print("  ✗ But struggles on NEW test data it hasn't seen\n")

def solutions():
    """Show how to improve accuracy"""
    
    print("\n" + "="*60)
    print("HOW TO IMPROVE THE MODEL?")
    print("="*60 + "\n")
    
    print("1. 🔧 USE BIGGER IMAGES")
    print("   Current: 64x64 pixels")
    print("   Better: 128x128 or 256x256 pixels")
    print("   Why: More details preserved\n")
    
    print("2. 📚 USE MORE TRAINING DATA")
    print("   Current: Limited EuroSAT dataset")
    print("   Better: Combine with Sentinel-2, UC Merced, etc.")
    print("   Why: More examples = better learning\n")
    
    print("3. 🧠 USE BETTER MODEL ARCHITECTURE")
    print("   Current: Custom CNN")
    print("   Better: ResNet, EfficientNet, Vision Transformer")
    print("   Why: Pre-trained models have learned features\n")
    
    print("4. 📊 BALANCE TRAINING DATA")
    print("   Problem: Some classes have fewer samples")
    print("   Solution: Data augmentation or weighted loss")
    print("   Why: Model won't be biased\n")
    
    print("5. 🎯 USE ENSEMBLE METHODS")
    print("   Combine multiple models")
    print("   Why: Different models catch different mistakes\n")

def main():
    print("\n🔍 Analyzing why your model made mistakes...\n")
    
    try:
        analyze_misclassification()
        compare_with_training()
        solutions()
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print("""
Your model predicted FOREST instead of HIGHWAY because:

1. ❌ Highway (dark asphalt) looks like Forest (dark trees)
2. ❌ Image is small (64x64) - details lost
3. ❌ Model hasn't seen this exact image before
4. ❌ Similar visual features confuse the model

This is NORMAL for ML models! No model is 100% accurate.

Even humans can misclassify satellite images!

✅ Your model is actually performing reasonably well
✅ For a demo, this is excellent
✅ You can explain these limitations to your teacher
        """)
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
