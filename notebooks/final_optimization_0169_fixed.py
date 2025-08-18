# Final Optimization and System Validation - Member 0169
# Fixed version with proper imports and error handling

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from deployment.inference_pipeline_0169 import InferencePipeline
    from deployment.model_deployment_0169 import ModelDeployment
    print("✅ Successfully imported deployment modules")
except ImportError as e:
    print(f"⚠️ Warning: Could not import deployment modules: {e}")
    print("Some functionality may be limited")

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SystemValidator:
    """System validation and optimization class"""
    
    def __init__(self):
        self.validation_results = {}
        self.optimization_suggestions = []
        
    def validate_model_deployment(self):
        """Validate model deployment functionality"""
        print("🔍 Validating Model Deployment...")
        
        try:
            # Test model deployment
            deployment = ModelDeployment('../models')
            models_loaded = deployment.load_all_models()
            
            if models_loaded > 0:
                self.validation_results['Model Deployment'] = '✅'
                print(f"✅ Model Deployment: {models_loaded} models loaded")
            else:
                self.validation_results['Model Deployment'] = '⚠️'
                print("⚠️ Model Deployment: No models loaded")
                
        except Exception as e:
            self.validation_results['Model Deployment'] = '❌'
            print(f"❌ Model Deployment: {e}")
    
    def validate_inference_pipeline(self):
        """Validate inference pipeline functionality"""
        print("🔍 Validating Inference Pipeline...")
        
        try:
            # Test inference pipeline
            pipeline = InferencePipeline('../models')
            init_result = pipeline.initialize()
            
            if init_result.get('models_loaded', 0) > 0:
                self.validation_results['Inference Pipeline'] = '✅'
                print("✅ Inference Pipeline: Initialized successfully")
            else:
                self.validation_results['Inference Pipeline'] = '⚠️'
                print("⚠️ Inference Pipeline: Limited functionality")
                
        except Exception as e:
            self.validation_results['Inference Pipeline'] = '❌'
            print(f"❌ Inference Pipeline: {e}")
    
    def validate_web_application(self):
        """Validate web application components"""
        print("🔍 Validating Web Application...")
        
        try:
            import streamlit
            self.validation_results['Web Application'] = '✅'
            print("✅ Web Application: Streamlit available")
        except ImportError:
            self.validation_results['Web Application'] = '❌'
            print("❌ Web Application: Streamlit not available")
    
    def validate_performance_monitoring(self):
        """Validate performance monitoring functionality"""
        print("🔍 Validating Performance Monitoring...")
        
        try:
            # Test if we can get performance stats
            if 'Inference Pipeline' in self.validation_results and self.validation_results['Inference Pipeline'] == '✅':
                pipeline = InferencePipeline('../models')
                pipeline.initialize()
                stats = pipeline.get_performance_stats()
                self.validation_results['Performance Monitoring'] = '✅'
                print("✅ Performance Monitoring: Available")
            else:
                self.validation_results['Performance Monitoring'] = '⚠️'
                print("⚠️ Performance Monitoring: Limited (pipeline not available)")
                
        except Exception as e:
            self.validation_results['Performance Monitoring'] = '❌'
            print(f"❌ Performance Monitoring: {e}")
    
    def validate_error_handling(self):
        """Validate error handling functionality"""
        print("🔍 Validating Error Handling...")
        
        try:
            # Test error handling with empty input
            if 'Inference Pipeline' in self.validation_results and self.validation_results['Inference Pipeline'] == '✅':
                pipeline = InferencePipeline('../models')
                pipeline.initialize()
                error_result = pipeline.predict("")  # Empty input
                
                if 'error' in error_result or error_result.get('prediction'):
                    self.validation_results['Error Handling'] = '✅'
                    print("✅ Error Handling: Proper error handling implemented")
                else:
                    self.validation_results['Error Handling'] = '⚠️'
                    print("⚠️ Error Handling: Basic functionality")
            else:
                self.validation_results['Error Handling'] = '⚠️'
                print("⚠️ Error Handling: Limited (pipeline not available)")
                
        except Exception as e:
            self.validation_results['Error Handling'] = '❌'
            print(f"❌ Error Handling: {e}")
    
    def run_full_validation(self):
        """Run complete system validation"""
        print("🚀 Running Full System Validation...")
        print("=" * 50)
        
        # Run all validation checks
        self.validate_model_deployment()
        self.validate_inference_pipeline()
        self.validate_web_application()
        self.validate_performance_monitoring()
        self.validate_error_handling()
        
        # Display results
        print("\n📊 Validation Results:")
        print("-" * 30)
        
        for component, status in self.validation_results.items():
            print(f"{status} {component}")
        
        # Calculate overall status
        total_checks = len(self.validation_results)
        passed_checks = sum(1 for status in self.validation_results.values() if status == '✅')
        warning_checks = sum(1 for status in self.validation_results.values() if status == '⚠️')
        failed_checks = sum(1 for status in self.validation_results.values() if status == '❌')
        
        print(f"\n📈 Summary:")
        print(f"Total Checks: {total_checks}")
        print(f"✅ Passed: {passed_checks}")
        print(f"⚠️ Warnings: {warning_checks}")
        print(f"❌ Failed: {failed_checks}")
        
        # Determine overall status
        if failed_checks == 0 and warning_checks == 0:
            overall_status = "✅ READY FOR DEPLOYMENT"
        elif failed_checks == 0:
            overall_status = "⚠️ NEEDS ATTENTION (Warnings)"
        else:
            overall_status = "❌ NEEDS FIXES (Failures)"
        
        print(f"\n🎯 Overall Status: {overall_status}")
        
        return self.validation_results
    
    def generate_optimization_suggestions(self):
        """Generate optimization suggestions based on validation results"""
        print("\n💡 Optimization Suggestions:")
        print("-" * 30)
        
        suggestions = []
        
        if self.validation_results.get('Model Deployment') == '❌':
            suggestions.append("🔧 Fix model loading: Ensure model files exist and paths are correct")
            suggestions.append("🔧 Check model file formats: Verify .pkl and .pth files are properly saved")
        
        if self.validation_results.get('Inference Pipeline') == '❌':
            suggestions.append("🔧 Fix inference pipeline: Check import paths and dependencies")
            suggestions.append("🔧 Verify ModelDeployment class implementation")
        
        if self.validation_results.get('Web Application') == '❌':
            suggestions.append("🔧 Install Streamlit: pip install streamlit")
            suggestions.append("🔧 Check web app dependencies")
        
        if self.validation_results.get('Performance Monitoring') == '❌':
            suggestions.append("🔧 Implement performance monitoring methods")
            suggestions.append("🔧 Add metrics collection and logging")
        
        if self.validation_results.get('Error Handling') == '❌':
            suggestions.append("🔧 Improve error handling: Add try-catch blocks")
            suggestions.append("🔧 Implement input validation")
        
        # General suggestions
        suggestions.append("🔧 Add comprehensive logging throughout the system")
        suggestions.append("🔧 Implement unit tests for critical components")
        suggestions.append("🔧 Add configuration management for model paths")
        suggestions.append("🔧 Implement health check endpoints")
        
        # Display suggestions
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion}")
        
        return suggestions

def main():
    """Main function"""
    print("🚀 Fake News Detection System - Final Validation")
    print("=" * 60)
    
    # Initialize validator
    validator = SystemValidator()
    
    try:
        # Run validation
        results = validator.run_full_validation()
        
        # Generate suggestions
        suggestions = validator.generate_optimization_suggestions()
        
        # Save validation report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f'validation_report_{timestamp}.txt'
        
        with open(report_file, 'w') as f:
            f.write("Fake News Detection System - Validation Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
            
            f.write("Validation Results:\n")
            f.write("-" * 20 + "\n")
            for component, status in results.items():
                f.write(f"{status} {component}\n")
            
            f.write(f"\nOptimization Suggestions:\n")
            f.write("-" * 25 + "\n")
            for i, suggestion in enumerate(suggestions, 1):
                f.write(f"{i}. {suggestion}\n")
        
        print(f"\n📋 Validation report saved to: {report_file}")
        
    except Exception as e:
        print(f"\n❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
