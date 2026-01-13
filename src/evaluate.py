import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, 
                            accuracy_score, precision_recall_fscore_support)
from tensorflow.keras.models import load_model
import os

class ModelEvaluator:
    def __init__(self, model_path, data_path='data/processed/preprocessed_data.npz'):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to trained model
            data_path: Path to preprocessed data
        """
        self.model = load_model(model_path)
        self.data_path = data_path
        
        # Load data
        self.load_data()
        
    def load_data(self):
        """Load test data"""
        print("Loading test data...")
        data = np.load(self.data_path, allow_pickle=True)
        
        self.X_test = data['X_test']
        self.y_test = data['y_test']
        self.classes = data['classes']
        
        print(f"Test samples: {len(self.X_test)}")
        print(f"Classes: {self.classes}")
        
    def evaluate(self):
        """Evaluate model on test set"""
        print("\nEvaluating model...")
        
        # Get predictions
        y_pred_probs = self.model.predict(self.X_test)
        self.y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Calculate metrics
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        self.precision, self.recall, self.f1, _ = precision_recall_fscore_support(
            self.y_test, self.y_pred, average='weighted'
        )
        
        print(f"\n{'='*60}")
        print("OVERALL METRICS")
        print(f"{'='*60}")
        print(f"Accuracy:  {self.accuracy:.4f} ({self.accuracy*100:.2f}%)")
        print(f"Precision: {self.precision:.4f}")
        print(f"Recall:    {self.recall:.4f}")
        print(f"F1-Score:  {self.f1:.4f}")
        
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1
        }
    
    def plot_confusion_matrix(self, save_path='results/confusion_matrix/'):
        """Plot confusion matrix"""
        os.makedirs(save_path, exist_ok=True)
        
        # Calculate confusion matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        # Plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.classes,
                   yticklabels=self.classes,
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(os.path.join(save_path, 'confusion_matrix.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Normalized confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=self.classes,
                   yticklabels=self.classes,
                   cbar_kws={'label': 'Percentage'})
        plt.title('Normalized Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(os.path.join(save_path, 'confusion_matrix_normalized.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nConfusion matrices saved to {save_path}")
        
    def plot_per_class_metrics(self, save_path='results/graphs/'):
        """Plot per-class precision, recall, and F1-score"""
        os.makedirs(save_path, exist_ok=True)
        
        # Calculate per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_test, self.y_pred, average=None
        )
        
        # Create DataFrame for easier plotting
        x = np.arange(len(self.classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(15, 6))
        
        bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8)
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Classes', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.classes, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'per_class_metrics.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print detailed classification report
        print(f"\n{'='*80}")
        print("DETAILED CLASSIFICATION REPORT")
        print(f"{'='*80}")
        print(classification_report(self.y_test, self.y_pred, 
                                   target_names=self.classes))
        
    def visualize_predictions(self, n_samples=15, save_path='results/predictions/'):
        """Visualize random predictions"""
        os.makedirs(save_path, exist_ok=True)
        
        # Random indices
        indices = np.random.choice(len(self.X_test), n_samples, replace=False)
        
        fig, axes = plt.subplots(3, 5, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            axes[i].imshow(self.X_test[idx])
            
            true_label = self.classes[self.y_test[idx]]
            pred_label = self.classes[self.y_pred[idx]]
            
            # Color based on correctness
            color = 'green' if self.y_test[idx] == self.y_pred[idx] else 'red'
            
            axes[i].set_title(f'True: {true_label}\nPred: {pred_label}', 
                            color=color, fontsize=9)
            axes[i].axis('off')
        
        plt.suptitle('Sample Predictions (Green=Correct, Red=Wrong)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'sample_predictions.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def analyze_errors(self, save_path='results/predictions/'):
        """Analyze misclassified samples"""
        os.makedirs(save_path, exist_ok=True)
        
        # Find misclassified samples
        misclassified_idx = np.where(self.y_test != self.y_pred)[0]
        
        print(f"\n{'='*60}")
        print(f"ERROR ANALYSIS")
        print(f"{'='*60}")
        print(f"Total misclassified: {len(misclassified_idx)} out of {len(self.y_test)}")
        print(f"Error rate: {len(misclassified_idx)/len(self.y_test)*100:.2f}%")
        
        # Visualize some misclassified samples
        if len(misclassified_idx) > 0:
            n_show = min(15, len(misclassified_idx))
            sample_errors = np.random.choice(misclassified_idx, n_show, replace=False)
            
            fig, axes = plt.subplots(3, 5, figsize=(15, 10))
            axes = axes.ravel()
            
            for i, idx in enumerate(sample_errors):
                axes[i].imshow(self.X_test[idx])
                
                true_label = self.classes[self.y_test[idx]]
                pred_label = self.classes[self.y_pred[idx]]
                
                axes[i].set_title(f'True: {true_label}\nPred: {pred_label}', 
                                color='red', fontsize=9)
                axes[i].axis('off')
            
            # Hide remaining subplots
            for i in range(n_show, 15):
                axes[i].axis('off')
            
            plt.suptitle('Misclassified Samples', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'misclassified_samples.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()


# Example usage
if __name__ == "__main__":
    # Create directories
    os.makedirs('results/confusion_matrix', exist_ok=True)
    os.makedirs('results/graphs', exist_ok=True)
    os.makedirs('results/predictions', exist_ok=True)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path='models/saved_models/best_model.h5'
    )
    
    # Run evaluation
    metrics = evaluator.evaluate()
    
    # Generate visualizations
    evaluator.plot_confusion_matrix()
    evaluator.plot_per_class_metrics()
    evaluator.visualize_predictions()
    evaluator.analyze_errors()
    
    print("\nEvaluation complete!")