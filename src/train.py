import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.model_architecture import LULCClassifier

class ModelTrainer:
    def __init__(self, model, data_path='data/processed/preprocessed_data.npz'):
        """
        Initialize trainer
        
        Args:
            model: Compiled Keras model
            data_path: Path to preprocessed data
        """
        self.model = model
        self.data_path = data_path
        self.history = None
        
        # Load data
        self.load_data()
        
    def load_data(self):
        """Load preprocessed data"""
        print("Loading preprocessed data...")
        data = np.load(self.data_path, allow_pickle=True)
        
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']
        self.X_test = data['X_test']
        self.y_test = data['y_test']
        self.classes = data['classes']
        
        print(f"Training samples: {len(self.X_train)}")
        print(f"Validation samples: {len(self.X_val)}")
        print(f"Test samples: {len(self.X_test)}")
        
    def setup_callbacks(self, model_save_path='models/saved_models/best_model.h5'):
        """Setup training callbacks"""
        
        # Create directory if not exists
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        callbacks = [
            # Save best model
            ModelCheckpoint(
                model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train_with_augmentation(self, epochs=50, batch_size=32):
        """
        Train model with data augmentation
        """
        print("\nStarting training with data augmentation...")
        
        # Data augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.2,
            fill_mode='nearest'
        )
        
        # Setup callbacks
        callbacks = self.setup_callbacks()
        
        # Train model
        self.history = self.model.fit(
            train_datagen.flow(self.X_train, self.y_train, batch_size=batch_size),
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\nTraining complete!")
        return self.history
    
    def train_without_augmentation(self, epochs=50, batch_size=32):
        """
        Train model without data augmentation
        """
        print("\nStarting training without data augmentation...")
        
        # Setup callbacks
        callbacks = self.setup_callbacks()
        
        # Train model
        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\nTraining complete!")
        return self.history
    
    def plot_training_history(self, save_path='results/graphs/'):
        """Plot training history"""
        if self.history is None:
            print("No training history available!")
            return
        
        os.makedirs(save_path, exist_ok=True)
        
        # Plot accuracy
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(self.history.history['accuracy'], label='Train Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Plot loss
        plt.subplot(1, 3, 2)
        plt.plot(self.history.history['loss'], label='Train Loss')
        plt.plot(self.history.history['val_loss'], label='Val Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot learning rate (if available)
        plt.subplot(1, 3, 3)
        if 'lr' in self.history.history:
            plt.plot(self.history.history['lr'])
            plt.title('Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'training_history.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nTraining plots saved to {save_path}")


# Example usage
if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('models/saved_models', exist_ok=True)
    os.makedirs('results/graphs', exist_ok=True)
    
    # Build and compile model
    classifier = LULCClassifier(input_shape=(64, 64, 3), num_classes=10)
    model = classifier.build_custom_cnn()
    model = classifier.compile_model(model, learning_rate=0.001)
    
    # Initialize trainer
    trainer = ModelTrainer(model)
    
    # Train model
    history = trainer.train_with_augmentation(epochs=50, batch_size=32)
    
    # Plot results
    trainer.plot_training_history()
    
    print("\nTraining completed successfully!")
    print("Best model saved to: models/saved_models/best_model.h5")