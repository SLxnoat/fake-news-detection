

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import pickle
import os

class MultiModalAttention(nn.Module):
    """Multi-modal attention mechanism for feature fusion"""
    
    def __init__(self, bert_dim, tfidf_dim, meta_dim, attention_dim=128):
        super(MultiModalAttention, self).__init__()
        
        # Project features to same dimension
        self.bert_proj = nn.Linear(bert_dim, attention_dim)
        self.tfidf_proj = nn.Linear(tfidf_dim, attention_dim)
        self.meta_proj = nn.Linear(meta_dim, attention_dim)
        
        # Attention weights
        self.attention = nn.Sequential(
            nn.Linear(attention_dim, attention_dim // 2),
            nn.ReLU(),
            nn.Linear(attention_dim // 2, 1)
        )
        
    def forward(self, bert_feat, tfidf_feat, meta_feat):
        # Project to same dimension
        bert_proj = self.bert_proj(bert_feat)
        tfidf_proj = self.tfidf_proj(tfidf_feat)
        meta_proj = self.meta_proj(meta_feat)
        
        # Stack features
        features = torch.stack([bert_proj, tfidf_proj, meta_proj], dim=1)  # [batch, 3, attention_dim]
        
        # Compute attention weights
        attention_weights = self.attention(features)  # [batch, 3, 1]
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Weighted combination
        attended_features = torch.sum(features * attention_weights, dim=1)
        
        return attended_features, attention_weights.squeeze(-1)

class HybridFakeNewsDetector(nn.Module):
    """
    Advanced Hybrid Fake News Detector
    Combines BERT, TF-IDF, and metadata with attention mechanism
    """
    
    def __init__(self, bert_dim=768, tfidf_dim=5000, meta_dim=10, n_classes=6, 
                 dropout_rate=0.3, use_attention=True):
        super(HybridFakeNewsDetector, self).__init__()
        
        self.use_attention = use_attention
        self.n_classes = n_classes
        
        # Feature processing layers
        self.bert_processor = nn.Sequential(
            nn.Linear(bert_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.tfidf_processor = nn.Sequential(
            nn.Linear(tfidf_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.meta_processor = nn.Sequential(
            nn.Linear(meta_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate // 2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        if use_attention:
            # Attention-based fusion
            self.attention_fusion = MultiModalAttention(256, 256, 32, attention_dim=128)
            fusion_input_dim = 128
        else:
            # Simple concatenation
            fusion_input_dim = 256 + 256 + 32
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, n_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, bert_features, tfidf_features, meta_features, return_attention=False):
        # Process each feature type
        bert_out = self.bert_processor(bert_features)
        tfidf_out = self.tfidf_processor(tfidf_features)
        meta_out = self.meta_processor(meta_features)
        
        if self.use_attention:
            # Attention-based fusion
            fused_features, attention_weights = self.attention_fusion(bert_out, tfidf_out, meta_out)
            output = self.classifier(fused_features)
            
            if return_attention:
                return output, attention_weights
            else:
                return output
        else:
            # Simple concatenation
            combined = torch.cat([bert_out, tfidf_out, meta_out], dim=1)
            output = self.classifier(combined)
            
            return output
    
    def get_feature_imrtance(self, dataloader, device):
        """Get feature importance using attention weights"""
        if not self.use_attention:
            print("⚠️ Feature importance only available with attention mechanism")
            return None
        
        self.eval()
        all_attention_weights = []
        
        with torch.no_grad():
            for batch in dataloader:
                bert_feat = batch['bert'].to(device)
                tfidf_feat = batch['tfidf'].to(device)
                meta_feat = batch['metadata'].to(device)
                
                _, attention_weights = self.forward(bert_feat, tfidf_feat, meta_feat, return_attention=True)
                all_attention_weights.append(attention_weights.cpu().numpy())
        
        attention_matrix = np.vstack(all_attention_weights)
        mean_attention = attention_matrix.mean(axis=0)
        
        feature_names = ['BERT', 'TF-IDF', 'Metadata']
        importance_dict = dict(zip(feature_names, mean_attention))
        
        return importance_dict

class HybridTrainer:
    """Training pipeline for the hybrid model"""
    
    def __init__(self, model, device='cpu', learning_rate=1e-3, weight_decay=1e-4):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=5, 
            gamma=0.7
        )
        
        self.train_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            # Move data to device
            bert_feat = batch['bert'].to(self.device)
            tfidf_feat = batch['tfidf'].to(self.device)
            meta_feat = batch['metadata'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(bert_feat, tfidf_feat, meta_feat)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                bert_feat = batch['bert'].to(self.device)
                tfidf_feat = batch['tfidf'].to(self.device)
                meta_feat = batch['metadata'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(bert_feat, tfidf_feat, meta_feat)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, curacy
    
    def train_model(self, train_loader, val_loader, epochs=15, save_path='../../models/best_hybrid_model.pth'):
        """Complete training loop"""
        best_val_acc = 0
        patience = 5
        patience_counter = 0
        
        print(f"🚀 Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Save history
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_acc)
            self.train_history['learning_rates'].append(current_lr)
            
            epoch_time = time.time() - start_time
            
            print(f"Epoch {epoch+1:2d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                  f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch':poch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'train_history': self.train_history
              }, save_path)
                print(f"💾 New best model saved! Val Acc: {best_val_acc:2f}%")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"⏹️ Early stopping triggered after {epoch+1} epochs")
                break
        
        print(f"✅ Training completed! Best validation accuracy: {best_val_acc:.2f}%")
        return self.train_history
    
    def plot_training_history(self, save_path='../../results/plots/training_history_0173.png'):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.train_history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(self.train_history['val_loss'], label='Validation Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[0, 1].plot(self.train_history['train_acc'], label='Train Accuracy', color='blue')
        axes[0, 1].plot(self.train_history['val_acc'], label='Validation Accuracy', color='red')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate plot
        axes[1, 0].plot(self.train_history['learning_rates'], color='green')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Loss difference
        loss_diff = np.array(self.train_history['val_loss']) - np.array(self.train_history['train_loss'])
        axes[1, 1].plot(loss_diff, color='purple')
        axes[1, 1].set_title('Overfitting Monitor (Va oss - Train Loss)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss Difference')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Training history saved to {save_path}")

def evaluate_model(model, test_loader, device, label_names=None):
    """Comprehensive model evaluation"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in test_loader:
            bert_feat = batch['bert'].to(device)
            tfidf_feat = batch['tfidf'].to(device)
            meta_feat = batch['metadata'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(bert_feat, tfidf_feat, meta_feat)
            probabilities = F.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    
    if label_names is None:
        lab_names = ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']
    
    # Classification report
    report = classification_report(all_labels, all_predictions, 
                                 target_names=label_names, 
                                 output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    print(f"🎯 Model Evaluation Results:")
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Macro F1-Score: {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1-Score: {report['weighted avg']['f1-score']:.4f}")
    
    return {
        'accuracy': accuracy,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': np.array(all_probabilities),
        'classification_report': report,
        'confusion_matrix': cm
    }

def plot_confusion_matrix(cm, label_names, save_path='../../results/plots/confusion_matr_0173.png'):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cm='Blues',
                xticklabels=label_names, yticklabels=label_names)
    plt.title('Confusion Matrix - Hybrid Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 Confusion matrix saved to {save_path}")

def test_hybrid_model():
    """Test hybrid model with sample data"""
    print("🧪 Testing Hybrid Model Architecture...")
    
    # Test with dummy data
    batch_size = 4
    bert_dim, tfidf_dim, meta_dim = 768, 5000, 10
    
    # Create dummy features
    bt_features = torch.randn(batch_size, bert_dim)
    tfidf_features = torch.randn(batch_size, tfidf_dim)
    meta_features = torch.randn(batch_size, meta_dim)
    
    # Test both attention and non-attention models
    models = {
        'with_attention': HybridFakeNewsDetector(use_attention=True),
        'without_attention': HybridFakeNewsDetector(use_attention=False)
    }
    
    for name, model in models.items():
        print(f"\n🔧 Testing {name} model:")
        
        # Forward pass
       if name == 'with_attention':
            outputs, attention = model(bert_features, tfidf_features, meta_features, return_attention=True)
            print(f"   Output shape: {outputs.shape}")
            print(f"   Attention weights shape: {attention.shape}")
            print(f"   Attention weights example: {attention[0].detach().numpy()}")
        else:
           outputs = model(bert_features, tfidf_features, meta_features)
            print(f"   Output shape: {outputs.shape}")
        
        # Model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
    
    print("✅ Hybrid model architecture test completed!")
    return models['with_attention']

class ModelExplainer:
    """Model interpretation and explanation utilities"""
    
    def __init__(self, model, feature_pipeline):
        self.model = model
        self.feature_pipeline = feature_pipeline
        
    def explain_prediction(self, statement, speaker="", party="", subject=""):
        """Explain a single prediction"""
        # Create sample dataframe
        sample_df = pd.DataFrame({
            'statement': [statement],
            'speaker': [speaker],
            'party_affiliation': [party],
            'subjects': [subject],
            'barely_true_counts': [0],
            'false_counts': [0],
            'half_true_counts': [0],
            'mostly_true_counts': [0],
            'pants_fire_counts': [0]
        })
        
        # Extract features
        features = self.feature_pipeline.transform(sample_df, "temp_explain", extract_bert=True)
        
        # Convert to tensors
        bert_tensor = torch.FloatTensor(features['bert'])
        tfidf_tensor = torch.FloatTensor(features['tfidf'])
        meta_tensor = torch.FloatTensor(features['metadata'])
        
        # Get prediction and attention
        self.model.eval()
        with torch.no_grad():
            if self.model.use_attention:
                output, attention = self.model(bert_tensor, tfidf_tensor, meta_tensor, return_attention=True)
                attention_weights = attention[0].numpy()
            else:
                output = self.model(bert_tensor, tfidf_tensor, meta_tensor)
                attention_weights = None
            
            probabilities = F.softmax(output, dim=1)[0].numpy()
            predicted_class = torch.argmax(output, dim=1)[0].item()
        
        # Map predictions
        lab_names = ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']
        prediction = label_names[predicted_class]
        confidence = probabilitis[predicted_class]
        
        result = {
            'prediction': prdiction,
            'confidence': confidence,
            'probabilities': dict(zip(label_names, probabilities)
            'attention_weights': attention_weights
        }
        
        return result

def main():
    """Main function for Day 2 testing"""
    print("🚀 Day 2: Hybrid ModelArchitecture - Member 0173")
    print("=" * 60)
    
    # Test 1: Model Architecture
    try:
        model = test_hybrid_model()
        print("✅ Model architecture test passed!")
    except Exception as e:
        print(f"❌ Model architecture test failed: {e}")
        return
   
    # Test 2: Training Pipeline (dry run)
    print("\n🧪 Testing Trning Pipeline...")
    try:
        device = torch.devic'cuda' if torch.cuda.is_available() else 'cpu
        trainer = HybridTrainer(model,evice=device)
        print(f"Training pipeline initialized on {vice}")
        print(f"   Optimizer: {type(trai .optimizer).__name__}")
        print(f"   Scheduler: {type(trainer.scheduler).__name__}")
        print(f"   Loss function: {type(trainer.criterion).__name__}")
    except Exception as e:
        print(f"❌ Training pipeline test failed: {e}")
        return
    
    print(f"\n🎯 Day 2 Architecture Implementation Complete!")
    print("✅ Multi-modal attention mechanism")
    print("✅ Hybrid model architecture") 
    print("✅ Training pipeline")
    print("✅ Evaluation framework")
    print("✅ Model explanation utilities")
    
    print(f"\n📋 Ready for Day 3:")
    print("- Train models with real data")
    print("- Hyperparameter optimization") 
    print("- Cross-validation")
    print("- Performance analysis")

if __name__ == "__main__":
    main()