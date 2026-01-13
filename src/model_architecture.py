import tensorflow as tf
from tensorflow.keras import layers, models, Model
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.regularizers import l2

class LULCClassifier:
    def __init__(self, input_shape=(64, 64, 3), num_classes=10):
        """
        Initialize LULC Classifier
        
        Args:
            input_shape: Input image shape
            num_classes: Number of land cover classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        
    def build_custom_cnn(self):
        """
        Build custom CNN architecture
        Designed specifically for satellite image classification
        """
        model = models.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', 
                         padding='same', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.4),
            
            # Block 4
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.4),
            
            # Fully connected layers
            layers.Flatten(),
            layers.Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build_resnet_transfer(self):
        """
        Build model using ResNet50 transfer learning
        """
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model
        base_model.trainable = False
        
        # Add custom head
        inputs = layers.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        
        return model
    
    def build_efficientnet_transfer(self):
        """
        Build model using EfficientNet transfer learning
        """
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model
        base_model.trainable = False
        
        # Add custom head
        inputs = layers.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        
        return model
    
    def compile_model(self, model, learning_rate=0.001):
        """
        Compile the model with optimizer and loss
        """
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def get_model_summary(self, model):
        """Print model architecture summary"""
        return model.summary()


# Example usage
if __name__ == "__main__":
    # Initialize classifier
    classifier = LULCClassifier(input_shape=(64, 64, 3), num_classes=10)
    
    # Build custom CNN
    print("=" * 60)
    print("CUSTOM CNN MODEL")
    print("=" * 60)
    model_cnn = classifier.build_custom_cnn()
    model_cnn = classifier.compile_model(model_cnn)
    model_cnn.summary()
    
    print("\n" + "=" * 60)
    print("MODEL PARAMETERS")
    print("=" * 60)
    
    total_params = model_cnn.count_params()
    print(f"Total parameters: {total_params:,}")
    
    # Calculate model size
    model_size_mb = (total_params * 4) / (1024 * 1024)  # Assuming float32
    print(f"Approximate model size: {model_size_mb:.2f} MB")