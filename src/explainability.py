import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Model, load_model
import tensorflow as tf
import os

class ExplainableAI:
    def __init__(self, model_path, data_path='data/processed/preprocessed_data.npz'):
        """
        Initialize Explainable AI module
        
        Args:
            model_path: Path to trained model
            data_path: Path to preprocessed data
        """
        self.model = load_model(model_path)
        self.data_path = data_path
        
        # Load data first
        self.load_data()
        
        # Build the model by calling it on actual data
        # This ensures model.inputs and model.outputs are defined
        if len(self.X_test) > 0:
            _ = self.model(self.X_test[:1])
        
    def load_data(self):
        """Load test data"""
        data = np.load(self.data_path, allow_pickle=True)
        self.X_test = data['X_test']
        self.y_test = data['y_test']
        self.classes = data['classes']
        
    def get_gradcam_heatmap(self, img_array, pred_index=None, last_conv_layer_name=None):
        """
        Generate Grad-CAM heatmap
        
        Args:
            img_array: Input image
            pred_index: Class index to visualize
            last_conv_layer_name: Name of last convolutional layer
        """
        # Find last convolutional layer if not specified
        if last_conv_layer_name is None:
            for layer in reversed(self.model.layers):
                if 'conv' in layer.name.lower():
                    last_conv_layer_name = layer.name
                    break
        
        # Get the last conv layer
        last_conv_layer = self.model.get_layer(last_conv_layer_name)
        
        # Create model that outputs feature maps and predictions
        # Use model.inputs instead of model.input for Sequential models
        grad_model = Model(
            inputs=self.model.inputs,
            outputs=[last_conv_layer.output, self.model.output]
        )
        
        # Compute gradient
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            
            class_channel = predictions[:, pred_index]
        
        # Gradient of class score with respect to feature map
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Pooled gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
    
    def overlay_heatmap(self, img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Overlay heatmap on original image
        
        Args:
            img: Original image
            heatmap: Grad-CAM heatmap
            alpha: Transparency
            colormap: OpenCV colormap
        """
        # Resize heatmap to match image size
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        # Convert heatmap to RGB
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, colormap)
        
        # Convert image to uint8
        img_uint8 = np.uint8(255 * img)
        
        # Superimpose heatmap
        superimposed = cv2.addWeighted(img_uint8, 1-alpha, heatmap, alpha, 0)
        
        return superimposed
    
    def visualize_gradcam(self, n_samples=6, save_path='results/explainability/'):
        """
        Visualize Grad-CAM for random samples
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Random samples
        indices = np.random.choice(len(self.X_test), n_samples, replace=False)
        
        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))
        
        for i, idx in enumerate(indices):
            img = self.X_test[idx:idx+1]
            
            # Get prediction
            pred = self.model.predict(img, verbose=0)
            pred_class = np.argmax(pred[0])
            true_class = self.y_test[idx]
            
            # Generate heatmap
            heatmap = self.get_gradcam_heatmap(img, pred_index=pred_class)
            
            # Overlay heatmap
            superimposed = self.overlay_heatmap(self.X_test[idx], heatmap)
            
            # Plot original image
            axes[i, 0].imshow(self.X_test[idx])
            axes[i, 0].set_title('Original Image', fontsize=10)
            axes[i, 0].axis('off')
            
            # Plot heatmap
            axes[i, 1].imshow(heatmap, cmap='jet')
            axes[i, 1].set_title('Grad-CAM Heatmap', fontsize=10)
            axes[i, 1].axis('off')
            
            # Plot overlay
            axes[i, 2].imshow(superimposed)
            color = 'green' if pred_class == true_class else 'red'
            axes[i, 2].set_title(
                f'True: {self.classes[true_class]}\nPred: {self.classes[pred_class]}',
                fontsize=10, color=color
            )
            axes[i, 2].axis('off')
        
        plt.suptitle('Grad-CAM Visualization', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'gradcam_visualization.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Grad-CAM visualizations saved to {save_path}")
    
    def visualize_feature_maps(self, img_idx=0, layer_indices=[1, 5, 10], 
                              save_path='results/explainability/'):
        """
        Visualize intermediate feature maps
        """
        os.makedirs(save_path, exist_ok=True)
        
        img = self.X_test[img_idx:img_idx+1]
        
        # Get layer outputs
        layer_outputs = [self.model.layers[i].output for i in layer_indices]
        activation_model = Model(inputs=self.model.input, outputs=layer_outputs)
        
        activations = activation_model.predict(img, verbose=0)
        
        # Visualize
        for layer_idx, activation in zip(layer_indices, activations):
            layer_name = self.model.layers[layer_idx].name
            n_features = min(16, activation.shape[-1])
            
            size = activation.shape[1]
            n_cols = 4
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
            axes = axes.flatten()
            
            for i in range(n_features):
                axes[i].imshow(activation[0, :, :, i], cmap='viridis')
                axes[i].axis('off')
                axes[i].set_title(f'Filter {i}')
            
            # Hide empty subplots
            for i in range(n_features, len(axes)):
                axes[i].axis('off')
            
            plt.suptitle(f'Feature Maps - {layer_name}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'feature_maps_{layer_name}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Feature maps saved to {save_path}")
    
    def analyze_model_attention(self, n_samples=5, save_path='results/explainability/'):
        """
        Analyze which regions the model focuses on
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Sample images from each class
        attention_analysis = {}
        
        for class_idx, class_name in enumerate(self.classes):
            # Find samples of this class
            class_samples = np.where(self.y_test == class_idx)[0]
            
            if len(class_samples) == 0:
                continue
            
            # Random sample
            sample_idx = np.random.choice(class_samples, 1)[0]
            img = self.X_test[sample_idx:sample_idx+1]
            
            # Get heatmap
            heatmap = self.get_gradcam_heatmap(img, pred_index=class_idx)
            
            attention_analysis[class_name] = {
                'image': self.X_test[sample_idx],
                'heatmap': heatmap
            }
        
        # Visualize
        n_classes = len(attention_analysis)
        fig, axes = plt.subplots(n_classes, 3, figsize=(12, 3*n_classes))
        
        if n_classes == 1:
            axes = axes.reshape(1, -1)
        
        for i, (class_name, data) in enumerate(attention_analysis.items()):
            img = data['image']
            heatmap = data['heatmap']
            overlay = self.overlay_heatmap(img, heatmap)
            
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f'{class_name} - Original')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(heatmap, cmap='jet')
            axes[i, 1].set_title(f'{class_name} - Attention')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(overlay)
            axes[i, 2].set_title(f'{class_name} - Overlay')
            axes[i, 2].axis('off')
        
        plt.suptitle('Model Attention per Class', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'class_attention_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Attention analysis saved to {save_path}")


# Example usage
if __name__ == "__main__":
    # Create directory
    os.makedirs('results/explainability', exist_ok=True)
    
    # Initialize explainer
    explainer = ExplainableAI(
        model_path='models/saved_models/best_model.h5'
    )
    
    # Generate Grad-CAM visualizations
    print("Generating Grad-CAM visualizations...")
    explainer.visualize_gradcam(n_samples=6)
    
    # Visualize feature maps
    print("\nVisualizing feature maps...")
    explainer.visualize_feature_maps(img_idx=0, layer_indices=[1, 5, 10])
    
    # Analyze attention
    print("\nAnalyzing model attention...")
    explainer.analyze_model_attention()
    
    print("\nExplainability analysis complete!")