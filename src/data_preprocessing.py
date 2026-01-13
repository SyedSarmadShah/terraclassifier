import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from PIL import Image

class SatelliteDataPreprocessor:
    def __init__(self, data_path, img_size=(64, 64)):
        """
        Initialize preprocessor
        
        Args:
            data_path: Path to dataset folder
            img_size: Target image size (height, width)
        """
        self.data_path = data_path
        self.img_size = img_size
        self.classes = []
        self.X = []
        self.y = []
        
    def load_dataset(self):
        """Load images from folder structure"""
        print("Loading dataset...")
        
        # Get class folders
        self.classes = sorted([d for d in os.listdir(self.data_path) 
                              if os.path.isdir(os.path.join(self.data_path, d))])
        
        print(f"Found {len(self.classes)} classes: {self.classes}")
        
        for idx, class_name in enumerate(self.classes):
            class_path = os.path.join(self.data_path, class_name)
            print(f"Loading class {idx+1}/{len(self.classes)}: {class_name}")
            
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                
                try:
                    # Read image
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    
                    # Convert BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Resize
                    img = cv2.resize(img, self.img_size)
                    
                    self.X.append(img)
                    self.y.append(idx)
                    
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    
        self.X = np.array(self.X, dtype='float32')
        self.y = np.array(self.y)
        
        print(f"\nDataset loaded: {len(self.X)} images")
        print(f"Image shape: {self.X[0].shape}")
        
        return self.X, self.y, self.classes
    
    def normalize_images(self, X):
        """Normalize pixel values to [0, 1]"""
        return X / 255.0
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1):
        """Split data into train, validation, and test sets"""
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Second split: train and val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
        )
        
        print(f"\nData split:")
        print(f"Train: {len(X_train)} images")
        print(f"Validation: {len(X_val)} images")
        print(f"Test: {len(X_test)} images")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_data_augmentation(self):
        """Create data augmentation pipeline"""
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.2,
            fill_mode='nearest'
        )
        return datagen
    
    def visualize_samples(self, X, y, classes, n_samples=15):
        """Visualize random samples from dataset"""
        plt.figure(figsize=(15, 10))
        
        indices = np.random.choice(len(X), n_samples, replace=False)
        
        for i, idx in enumerate(indices):
            plt.subplot(3, 5, i + 1)
            plt.imshow(X[idx].astype('uint8'))
            plt.title(f"{classes[y[idx]]}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('results/sample_images.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    def visualize_augmentation(self, X, y, datagen, n_samples=5):
        """Visualize augmented images"""
        # Take one sample image (unnormalized for visualization)
        sample_idx = np.random.randint(0, len(X))
        sample_img = X[sample_idx:sample_idx+1]
        # Scale back to 0-255 for augmentation display
        sample_img = (sample_img * 255).astype('uint8')
        
        plt.figure(figsize=(15, 3))
        
        # Original
        plt.subplot(1, n_samples+1, 1)
        plt.imshow(sample_img[0].astype('uint8'))
        plt.title('Original')
        plt.axis('off')
        
        # Augmented versions
        i = 2
        for batch in datagen.flow(sample_img, batch_size=1):
            plt.subplot(1, n_samples+1, i)
            plt.imshow(batch[0].astype('uint8'))
            plt.title(f'Augmented {i-1}')
            plt.axis('off')
            i += 1
            if i > n_samples+1:
                break
        
        plt.tight_layout()
        plt.savefig('results/augmented_samples.png', dpi=150, bbox_inches='tight')
        plt.close()


# Example usage
if __name__ == "__main__":
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = SatelliteDataPreprocessor(
        data_path='data/raw/EuroSAT',  # Adjust path
        img_size=(64, 64)
    )
    
    # Load dataset
    X, y, classes = preprocessor.load_dataset()
    
    # Visualize samples
    preprocessor.visualize_samples(X, y, classes)
    
    # Normalize
    X_normalized = preprocessor.normalize_images(X)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        X_normalized, y
    )
    
    # Save preprocessed data
    np.savez_compressed(
        'data/processed/preprocessed_data.npz',
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        classes=classes
    )
    
    print("\nPreprocessing complete! Data saved.")