#!/usr/bin/env python3
"""
LULC Classification - Image Prediction Script
Simple script to classify any satellite image using the trained model
Perfect for demonstrations and presentations
"""

import os
import sys

# Check if running in virtual environment, if not try to activate it
if 'lulc_env' not in sys.prefix:
    print("⚠️  Using system Python instead of virtual environment!")
    print("Please run: source lulc_env/bin/activate")
    print("Then run: python predict_image.py\n")

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

class ImageClassifier:
    def __init__(self, model_path='models/saved_models/best_model.h5'):
        """Load the trained model"""
        print("Loading trained model...")
        self.model = load_model(model_path)
        self.img_size = (64, 64)
        
        # Land cover classes
        self.classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 
                       'Highway', 'Industrial', 'Pasture', 
                       'PermanentCrop', 'Residential', 'River', 'SeaLake']
        
        print("✓ Model loaded successfully!")
        print(f"Classes: {', '.join(self.classes)}\n")
    
    def preprocess_image(self, image_path):
        """Load and preprocess image"""
        # Read image
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"❌ Error: Could not load image from {image_path}")
            return None
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        img_resized = cv2.resize(img_rgb, self.img_size)
        
        # Normalize to [0, 1]
        img_normalized = img_resized.astype('float32') / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_resized, img_batch
    
    def predict(self, image_path):
        """Predict class and confidence for image"""
        result = self.preprocess_image(image_path)
        if result is None:
            return None
        
        img_resized, img_batch = result
        
        # Get prediction
        print("🔄 Classifying image...")
        predictions = self.model.predict(img_batch, verbose=0)
        
        # Get class and confidence
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = self.classes[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]
        
        return {
            'image': img_resized,
            'class': predicted_class,
            'confidence': confidence,
            'all_predictions': predictions[0],
            'image_path': image_path
        }
    
    def visualize_prediction(self, result, save_path=None):
        """Show prediction result with confidence"""
        if result is None:
            print("❌ No result to visualize")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Display image
        axes[0].imshow(result['image'])
        axes[0].set_title(f"Input Image", fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Display predictions as bar chart
        colors = ['green' if i == np.argmax(result['all_predictions']) else 'skyblue' 
                 for i in range(len(self.classes))]
        
        axes[1].barh(self.classes, result['all_predictions'], color=colors)
        axes[1].set_xlabel('Confidence Score', fontsize=11)
        axes[1].set_title('Prediction Confidence by Class', fontsize=12, fontweight='bold')
        axes[1].set_xlim(0, 1)
        
        # Add value labels
        for i, v in enumerate(result['all_predictions']):
            axes[1].text(v + 0.02, i, f'{v:.4f}', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Visualization saved: {save_path}")
        
        plt.show()
        
        return fig
    
    def print_result(self, result):
        """Print prediction result in formatted way"""
        if result is None:
            print("❌ No result to display")
            return
        
        print("\n" + "="*60)
        print("CLASSIFICATION RESULT")
        print("="*60)
        print(f"Image: {result['image_path']}")
        print(f"\n🎯 Predicted Class: {result['class']}")
        print(f"📊 Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
        print("\n" + "-"*60)
        print("Confidence Scores for All Classes:")
        print("-"*60)
        
        # Sort by confidence
        sorted_indices = np.argsort(result['all_predictions'])[::-1]
        
        for idx in sorted_indices:
            class_name = self.classes[idx]
            confidence = result['all_predictions'][idx]
            bar_length = int(confidence * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            print(f"{class_name:20} {confidence:.4f} {bar}")
        
        print("="*60 + "\n")

def main():
    """Main function"""
    print("\n" + "="*60)
    print("LULC CLASSIFICATION - IMAGE PREDICTION DEMO")
    print("="*60 + "\n")
    
    # Initialize classifier
    classifier = ImageClassifier()
    
    while True:
        print("Options:")
        print("1. Classify an image")
        print("2. Classify multiple images from a folder")
        print("3. Test with sample images")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            image_path = input("\nEnter image path: ").strip()
            
            if not os.path.exists(image_path):
                print(f"❌ File not found: {image_path}")
                continue
            
            result = classifier.predict(image_path)
            if result:
                classifier.print_result(result)
                
                # Ask to visualize
                viz = input("Show visualization? (y/n): ").strip().lower()
                if viz == 'y':
                    save = input("Save visualization? (y/n): ").strip().lower()
                    save_path = None
                    if save == 'y':
                        save_path = f"results/predictions/{os.path.basename(image_path)}_result.png"
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    classifier.visualize_prediction(result, save_path)
        
        elif choice == '2':
            folder_path = input("\nEnter folder path: ").strip()
            
            if not os.path.isdir(folder_path):
                print(f"❌ Folder not found: {folder_path}")
                continue
            
            image_files = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            if not image_files:
                print("❌ No image files found in folder")
                continue
            
            print(f"\nFound {len(image_files)} images. Processing...\n")
            
            for img_file in image_files[:10]:  # Limit to 10 for demo
                img_path = os.path.join(folder_path, img_file)
                result = classifier.predict(img_path)
                if result:
                    print(f"✓ {img_file}: {result['class']} ({result['confidence']:.2%})")
        
        elif choice == '3':
            # Test with actual test images from the preprocessed data
            print("\nLoading test images from preprocessed data...")
            
            data_path = 'data/processed/preprocessed_data.npz'
            if not os.path.exists(data_path):
                print("❌ Preprocessed data not found. Run main.py first.")
                continue
            
            data = np.load(data_path, allow_pickle=True)
            X_test = data['X_test']
            y_test = data['y_test']
            classes = data['classes']
            
            # Get random samples
            num_samples = 5
            indices = np.random.choice(len(X_test), num_samples, replace=False)
            
            print(f"\nTesting on {num_samples} random test images:\n")
            
            correct = 0
            for i, idx in enumerate(indices):
                img = X_test[idx]
                true_label_idx = np.argmax(y_test[idx])
                true_label = classes[true_label_idx]
                
                # Predict
                img_batch = np.expand_dims(img, axis=0)
                predictions = classifier.model.predict(img_batch, verbose=0)
                pred_idx = np.argmax(predictions[0])
                pred_label = classifier.classes[pred_idx]
                confidence = predictions[0][pred_idx]
                
                is_correct = pred_label == true_label
                if is_correct:
                    correct += 1
                
                status = "✓ CORRECT" if is_correct else "✗ WRONG"
                print(f"Sample {i+1}: True={true_label}, Predicted={pred_label} ({confidence:.2%}) {status}")
            
            print(f"\nAccuracy on sample: {correct}/{num_samples} ({correct/num_samples*100:.1f}%)")
        
        elif choice == '4':
            print("\nGoodbye! 👋")
            break
        
        else:
            print("❌ Invalid choice!")

if __name__ == "__main__":
    main()
