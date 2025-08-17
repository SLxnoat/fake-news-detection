import time
import psutil
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns

from .model_deployment_0169 import ModelDeployment
from .inference_pipeline_0169 import InferencePipeline

class PerformanceOptimizer:
    def __init__(self):
        self.benchmark_results = {}
        self.optimization_history = []
    
    def benchmark_inference_speed(self, pipeline, test_statements, iterations=50):
        """Benchmark inference speed"""
        print(f"Benchmarking inference speed with {iterations} iterations...")
        
        response_times = []
        memory_usage = []
        cpu_usage = []
        
        # Warm-up runs
        for _ in range(5):
            pipeline.predict(test_statements[0]['statement'])
        
        # Benchmark runs
        process = psutil.Process(os.getpid())
        
        for i in range(iterations):
            # Monitor system resources
            cpu_before = process.cpu_percent()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Time the prediction
            start_time = time.time()
            result = pipeline.predict(
                test_statements[i % len(test_statements)]['statement'],
                use_ensemble=True
            )
            end_time = time.time()
            
            response_time = end_time - start_time
            response_times.append(response_time)
            
            # Monitor system resources after
            cpu_after = process.cpu_percent()
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            
            cpu_usage.append(max(cpu_after, cpu_before))
            memory_usage.append(memory_after)
            
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{iterations} benchmarks")
        
        # Calculate statistics
        stats = {
            'iterations': iterations,
            'mean_response_time': np.mean(response_times),
            'median_response_time': np.median(response_times),
            'p95_response_time': np.percentile(response_times, 95),
            'p99_response_time': np.percentile(response_times, 99),
            'min_response_time': np.min(response_times),
            'max_response_time': np.max(response_times),
            'std_response_time': np.std(response_times),
            'mean_memory_mb': np.mean(memory_usage),
            'max_memory_mb': np.max(memory_usage),
            'mean_cpu_percent': np.mean(cpu_usage),
            'max_cpu_percent': np.max(cpu_usage),
            'throughput_rps': 1 / np.mean(response_times)
        }
        
        self.benchmark_results['inference_speed'] = {
            'stats': stats,
            'raw_data': {
                'response_times': response_times,
                'memory_usage': memory_usage,
                'cpu_usage': cpu_usage
            }
        }
        
        return stats
    
    def benchmark_batch_processing(self, pipeline, test_statements, batch_sizes=[1, 5, 10, 20]):
        """Benchmark batch processing performance"""
        print("Benchmarking batch processing...")
        
        batch_results = {}
        
        for batch_size in batch_sizes:
            print(f"Testing batch size: {batch_size}")
            
            # Create batch
            batch = test_statements[:batch_size] * (batch_size // len(test_statements) + 1)
            batch = batch[:batch_size]
            
            # Time batch processing
            start_time = time.time()
            results = pipeline.batch_predict(batch, use_ensemble=True)
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_time_per_item = total_time / batch_size
            throughput = batch_size / total_time
            
            batch_results[batch_size] = {
                'total_time': total_time,
                'avg_time_per_item': avg_time_per_item,
                'throughput': throughput,
                'successful_predictions': sum(1 for r in results if 'error' not in r)
            }
        
        self.benchmark_results['batch_processing'] = batch_results
        return batch_results
    
    def optimize_memory_usage(self, pipeline):
        """Analyze and optimize memory usage"""
        print("Analyzing memory usage...")
        
        process = psutil.Process(os.getpid())
        
        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Memory after loading models
        model_info = pipeline.deployment.get_model_info()
        models_memory = process.memory_info().rss / 1024 / 1024 - baseline_memory
        
        # Memory during prediction
        test_statement = "This is a test statement for memory analysis."
        memory_before = process.memory_info().rss / 1024 / 1024
        
        for _ in range(10):
            pipeline.predict(test_statement)
        
        memory_after = process.memory_info().rss / 1024 / 1024
        prediction_memory_overhead = memory_after - memory_before
        
        memory_analysis = {
            'baseline_memory_mb': baseline_memory,
            'models_memory_mb': models_memory,
            'prediction_overhead_mb': prediction_memory_overhead,
            'total_memory_mb': memory_after,
            'loaded_models': model_info['loaded_models']
        }
        
        self.benchmark_results['memory_analysis'] = memory_analysis
        
        # Memory optimization recommendations
        recommendations = []
        
        if models_memory > 500:  # If models take more than 500MB
            recommendations.append("Consider using model quantization to reduce memory usage")
        
        if prediction_memory_overhead > 50:  # If prediction overhead > 50MB
            recommendations.append("Optimize preprocessing pipeline to reduce memory allocation")
        
        memory_analysis['recommendations'] = recommendations
        
        return memory_analysis
    
    def stress_test(self, pipeline, test_statements, duration_minutes=5):
        """Perform stress testing"""
        print(f"Starting stress test for {duration_minutes} minutes...")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        predictions_made = 0
        successful_predictions = 0
        failed_predictions = 0
        response_times = []
        error_types = {}
        
        while time.time() < end_time:
            statement = test_statements[predictions_made % len(test_statements)]
            
            prediction_start = time.time()
            result = pipeline.predict(statement['statement'], use_ensemble=True)
            prediction_end = time.time()
            
            response_times.append(prediction_end - prediction_start)
            predictions_made += 1
            
            if 'error' in result:
                failed_predictions += 1
                error_type = type(result.get('error', 'Unknown')).__name__
                error_types[error_type] = error_types.get(error_type, 0) + 1
            else:
                successful_predictions += 1
            
            if predictions_made % 100 == 0:
                elapsed = time.time() - start_time
                remaining = (end_time - time.time()) / 60
                print(f"Predictions made: {predictions_made}, "
                      f"Success rate: {successful_predictions/predictions_made:.3f}, "
                      f"Remaining: {remaining:.1f} min")
        
        actual_duration = (time.time() - start_time) / 60
        
        stress_results = {
            'duration_minutes': actual_duration,
            'total_predictions': predictions_made,
            'successful_predictions': successful_predictions,
            'failed_predictions': failed_predictions,
            'success_rate': successful_predictions / predictions_made,
            'average_throughput': predictions_made / (actual_duration * 60),
            'mean_response_time': np.mean(response_times),
            'p95_response_time': np.percentile(response_times, 95),
            'error_types': error_types
        }
        
        self.benchmark_results['stress_test'] = stress_results
        return stress_results
    
    def generate_performance_report(self, save_path=None):
        """Generate comprehensive performance report"""
        if not self.benchmark_results:
            print("No benchmark results available")
            return
        
        report = "# Performance Optimization Report\n\n"
        report += f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Inference Speed Results
        if 'inference_speed' in self.benchmark_results:
            stats = self.benchmark_results['inference_speed']['stats']
            report += "## Inference Speed Benchmark\n\n"
            report += f"- **Iterations**: {stats['iterations']}\n"
            report += f"- **Mean Response Time**: {stats['mean_response_time']:.4f}s\n"
            report += f"- **Median Response Time**: {stats['median_response_time']:.4f}s\n"
            report += f"- **95th Percentile**: {stats['p95_response_time']:.4f}s\n"
            report += f"- **99th Percentile**: {stats['p99_response_time']:.4f}s\n"
            report += f"- **Throughput**: {stats['throughput_rps']:.2f} requests/second\n"
            report += f"- **Memory Usage**: {stats['mean_memory_mb']:.2f} MB (avg), {stats['max_memory_mb']:.2f} MB (max)\n\n"
        
        # Batch Processing Results
        if 'batch_processing' in self.benchmark_results:
            report += "## Batch Processing Performance\n\n"
            batch_data = self.benchmark_results['batch_processing']
            for batch_size, results in batch_data.items():
                report += f"**Batch Size {batch_size}:**\n"
                report += f"- Total Time: {results['total_time']:.4f}s\n"
                report += f"- Avg Time per Item: {results['avg_time_per_item']:.4f}s\n"
                report += f"- Throughput: {results['throughput']:.2f} items/second\n\n"
        
        # Memory Analysis
        if 'memory_analysis' in self.benchmark_results:
            memory = self.benchmark_results['memory_analysis']
            report += "## Memory Usage Analysis\n\n"
            report += f"- **Baseline Memory**: {memory['baseline_memory_mb']:.2f} MB\n"
            report += f"- **Models Memory**: {memory['models_memory_mb']:.2f} MB\n"
            report += f"- **Prediction Overhead**: {memory['prediction_overhead_mb']:.2f} MB\n"
            report += f"- **Total Memory**: {memory['total_memory_mb']:.2f} MB\n"
            
            if memory['recommendations']:
                report += "\n**Optimization Recommendations:**\n"
                for rec in memory['recommendations']:
                    report += f"- {rec}\n"
            report += "\n"
        
        # Stress Test Results
        if 'stress_test' in self.benchmark_results:
            stress = self.benchmark_results['stress_test']
            report += "## Stress Test Results\n\n"
            report += f"- **Duration**: {stress['duration_minutes']:.2f} minutes\n"
            report += f"- **Total Predictions**: {stress['total_predictions']}\n"
            report += f"- **Success Rate**: {stress['success_rate']:.3f}\n"
            report += f"- **Average Throughput**: {stress['average_throughput']:.2f} predictions/second\n"
            report += f"- **Mean Response Time**: {stress['mean_response_time']:.4f}s\n"
            report += f"- **95th Percentile Response Time**: {stress['p95_response_time']:.4f}s\n"
            
            if stress['error_types']:
                report += "\n**Error Types:**\n"
                for error_type, count in stress['error_types'].items():
                    report += f"- {error_type}: {count}\n"
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Performance report saved to {save_path}")
        
        return report
    
    def plot_performance_metrics(self, save_dir='../results/plots'):
        """Generate performance visualization plots"""
        if not self.benchmark_results:
            print("No benchmark results to plot")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Response time distribution
        if 'inference_speed' in self.benchmark_results:
            response_times = self.benchmark_results['inference_speed']['raw_data']['response_times']
            
            plt.figure(figsize=(12, 8))
            
            # Histogram
            plt.subplot(2, 2, 1)
            plt.hist(response_times, bins=30, alpha=0.7, color='skyblue')
            plt.xlabel('Response Time (seconds)')
            plt.ylabel('Frequency')
            plt.title('Response Time Distribution')
            
            # Box plot
            plt.subplot(2, 2, 2)
            plt.boxplot(response_times)
            plt.ylabel('Response Time (seconds)')
            plt.title('Response Time Box Plot')
            
            # Time series
            plt.subplot(2, 2, 3)
            plt.plot(response_times, alpha=0.7)
            plt.xlabel('Request Number')
            plt.ylabel('Response Time (seconds)')
            plt.title('Response Time Over Time')
            
            # Memory usage
            memory_usage = self.benchmark_results['inference_speed']['raw_data']['memory_usage']
            plt.subplot(2, 2, 4)
            plt.plot(memory_usage, color='orange', alpha=0.7)
            plt.xlabel('Request Number')
            plt.ylabel('Memory Usage (MB)')
            plt.title('Memory Usage Over Time')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
            plt.show()
        
        # Batch processing comparison
        if 'batch_processing' in self.benchmark_results:
            batch_data = self.benchmark_results['batch_processing']
            
            batch_sizes = list(batch_data.keys())
            throughputs = [batch_data[size]['throughput'] for size in batch_sizes]
            avg_times = [batch_data[size]['avg_time_per_item'] for size in batch_sizes]
            
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.bar(batch_sizes, throughputs, alpha=0.7, color='green')
            plt.xlabel('Batch Size')
            plt.ylabel('Throughput (items/second)')
            plt.title('Throughput vs Batch Size')
            
            plt.subplot(1, 2, 2)
            plt.bar(batch_sizes, avg_times, alpha=0.7, color='coral')
            plt.xlabel('Batch Size')
            plt.ylabel('Avg Time per Item (seconds)')
            plt.title('Average Processing Time vs Batch Size')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'batch_performance.png'), dpi=300, bbox_inches='tight')
            plt.show()