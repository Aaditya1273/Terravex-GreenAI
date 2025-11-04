"""
Edge Runner for Raspberry Pi and Low-Power Devices
Optimized green AI inference for resource-constrained environments
"""

import torch
import time
import psutil
import json
import os
from datetime import datetime
import numpy as np
from codecarbon import EmissionsTracker
import platform
import subprocess

class EdgeGreenAI:
    """Green AI runner optimized for edge devices"""
    
    def __init__(self, device_type='raspberry_pi'):
        self.device_type = device_type
        self.device_info = self._get_device_info()
        self.power_profile = self._get_power_profile()
        
    def _get_device_info(self):
        """Get detailed device information"""
        info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'architecture': platform.architecture(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'timestamp': datetime.now().isoformat()
        }
        
        # Try to get Raspberry Pi specific info
        try:
            if os.path.exists('/proc/cpuinfo'):
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read()
                    if 'Raspberry Pi' in cpuinfo:
                        info['device_model'] = 'Raspberry Pi'
                        # Extract model info
                        for line in cpuinfo.split('\n'):
                            if 'Model' in line:
                                info['model_details'] = line.split(':')[1].strip()
                                break
        except Exception:
            pass
            
        # Check for ARM architecture
        if 'arm' in platform.machine().lower() or 'aarch64' in platform.machine().lower():
            info['architecture_type'] = 'ARM'
        else:
            info['architecture_type'] = 'x86'
            
        return info
    
    def _get_power_profile(self):
        """Get power consumption profile for device type"""
        profiles = {
            'raspberry_pi': {
                'idle_watts': 2.5,
                'cpu_load_watts': 6.5,
                'max_watts': 8.0,
                'thermal_limit_celsius': 80,
                'recommended_batch_size': 1
            },
            'jetson_nano': {
                'idle_watts': 5.0,
                'cpu_load_watts': 10.0,
                'max_watts': 15.0,
                'thermal_limit_celsius': 85,
                'recommended_batch_size': 4
            },
            'laptop': {
                'idle_watts': 15.0,
                'cpu_load_watts': 45.0,
                'max_watts': 65.0,
                'thermal_limit_celsius': 90,
                'recommended_batch_size': 8
            },
            'desktop': {
                'idle_watts': 50.0,
                'cpu_load_watts': 150.0,
                'max_watts': 300.0,
                'thermal_limit_celsius': 85,
                'recommended_batch_size': 16
            }
        }
        
        return profiles.get(self.device_type, profiles['raspberry_pi'])
    
    def optimize_for_edge(self, model):
        """Apply edge-specific optimizations"""
        print(f"🔧 Optimizing model for {self.device_type}...")
        
        # Apply quantization
        model.eval()
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
        )
        
        # Additional ARM optimizations
        if self.device_info.get('architecture_type') == 'ARM':
            print("🔧 Applying ARM-specific optimizations...")
            # Set number of threads for ARM
            torch.set_num_threads(min(2, psutil.cpu_count()))
        
        return quantized_model
    
    def monitor_system_resources(self):
        """Monitor system resources during inference"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_mb': psutil.virtual_memory().available / (1024**2),
            'temperature_celsius': self._get_cpu_temperature(),
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_cpu_temperature(self):
        """Get CPU temperature (Raspberry Pi specific)"""
        try:
            # Try Raspberry Pi method
            if os.path.exists('/sys/class/thermal/thermal_zone0/temp'):
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    temp = int(f.read().strip()) / 1000.0
                    return temp
            
            # Try other methods
            try:
                result = subprocess.run(['vcgencmd', 'measure_temp'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    temp_str = result.stdout.strip()
                    temp = float(temp_str.split('=')[1].split("'")[0])
                    return temp
            except:
                pass
                
        except Exception:
            pass
        
        return None
    
    def run_edge_benchmark(self, model, test_data, num_samples=100):
        """Run benchmark optimized for edge devices"""
        print(f"🌍 Running Terravex edge benchmark on {self.device_type}")
        print(f"📊 Device Info: {self.device_info.get('model_details', 'Unknown')}")
        
        # Optimize model for edge
        optimized_model = self.optimize_for_edge(model)
        
        # Start emissions tracking
        tracker = EmissionsTracker(
            project_name=f"edge_green_ai_{self.device_type}",
            output_dir="./carbon_logs",
            log_level="ERROR"
        )
        
        tracker.start()
        start_time = time.time()
        
        # Collect system metrics
        system_metrics = []
        inference_times = []
        
        print(f"🔄 Running {num_samples} inferences...")
        
        optimized_model.eval()
        with torch.no_grad():
            for i in range(min(num_samples, len(test_data))):
                # Monitor system before inference
                pre_metrics = self.monitor_system_resources()
                
                # Check thermal throttling
                temp = pre_metrics.get('temperature_celsius')
                if temp and temp > self.power_profile['thermal_limit_celsius']:
                    print(f"🌡️ Thermal throttling detected: {temp:.1f}°C")
                    time.sleep(2)  # Cool down
                
                # Run inference
                inference_start = time.time()
                
                if hasattr(test_data, '__getitem__'):
                    sample = test_data[i]
                    if isinstance(sample, tuple):
                        input_data = sample[0].unsqueeze(0) if len(sample[0].shape) == 3 else sample[0]
                    else:
                        input_data = sample.unsqueeze(0) if len(sample.shape) == 3 else sample
                else:
                    # Handle DataLoader
                    for j, batch in enumerate(test_data):
                        if j == i:
                            input_data = batch[0][:1] if isinstance(batch, tuple) else batch[:1]
                            break
                
                output = optimized_model(input_data)
                inference_end = time.time()
                
                inference_time = inference_end - inference_start
                inference_times.append(inference_time)
                
                # Monitor system after inference
                post_metrics = self.monitor_system_resources()
                system_metrics.append({
                    'inference_id': i,
                    'inference_time_ms': inference_time * 1000,
                    'pre_inference': pre_metrics,
                    'post_inference': post_metrics
                })
                
                # Progress update
                if (i + 1) % 10 == 0:
                    avg_time = np.mean(inference_times[-10:]) * 1000
                    print(f"   Completed {i+1}/{num_samples} - Avg time: {avg_time:.1f}ms")
        
        end_time = time.time()
        emissions = tracker.stop()
        
        # Calculate metrics
        total_time = end_time - start_time
        avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
        throughput = num_samples / total_time  # Inferences per second
        
        # System resource analysis
        avg_cpu = np.mean([m['post_inference']['cpu_percent'] for m in system_metrics])
        avg_memory = np.mean([m['post_inference']['memory_percent'] for m in system_metrics])
        
        temperatures = [m['post_inference']['temperature_celsius'] 
                       for m in system_metrics 
                       if m['post_inference']['temperature_celsius'] is not None]
        avg_temp = np.mean(temperatures) if temperatures else None
        max_temp = np.max(temperatures) if temperatures else None
        
        results = {
            'device_info': self.device_info,
            'power_profile': self.power_profile,
            'benchmark_results': {
                'total_inferences': num_samples,
                'total_time_seconds': total_time,
                'avg_inference_time_ms': avg_inference_time,
                'throughput_inferences_per_second': throughput,
                'energy_consumed_kwh': emissions.energy_consumed if emissions else None,
                'co2_emissions_kg': emissions.emissions if emissions else None
            },
            'system_performance': {
                'avg_cpu_percent': avg_cpu,
                'avg_memory_percent': avg_memory,
                'avg_temperature_celsius': avg_temp,
                'max_temperature_celsius': max_temp,
                'thermal_throttling_detected': max_temp > self.power_profile['thermal_limit_celsius'] if max_temp else False
            },
            'detailed_metrics': system_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        results_filename = f"edge_benchmark_{self.device_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print summary
        self._print_edge_summary(results)
        
        return results
    
    def _print_edge_summary(self, results):
        """Print edge benchmark summary"""
        print(f"\n{'='*60}")
        print(f"🌍 TERRAVEX EDGE AI BENCHMARK RESULTS")
        print(f"{'='*60}")
        
        device_info = results['device_info']
        benchmark = results['benchmark_results']
        system = results['system_performance']
        
        print(f"🔧 Device: {device_info.get('model_details', self.device_type)}")
        print(f"🏗️ Architecture: {device_info.get('architecture_type', 'Unknown')}")
        print(f"💾 Memory: {device_info.get('memory_total_gb', 0):.1f} GB")
        
        print(f"\n⚡ Performance:")
        print(f"   Avg Inference Time: {benchmark['avg_inference_time_ms']:.1f} ms")
        print(f"   Throughput: {benchmark['throughput_inferences_per_second']:.2f} inferences/sec")
        print(f"   Total Inferences: {benchmark['total_inferences']}")
        
        print(f"\n🌱 Energy & Emissions:")
        if benchmark['energy_consumed_kwh']:
            print(f"   Energy Consumed: {benchmark['energy_consumed_kwh']:.6f} kWh")
            print(f"   CO₂ Emissions: {benchmark['co2_emissions_kg']:.6f} kg")
            
            # Calculate per-inference metrics
            energy_per_inference = benchmark['energy_consumed_kwh'] / benchmark['total_inferences'] * 1000000  # Convert to µWh
            co2_per_inference = benchmark['co2_emissions_kg'] / benchmark['total_inferences'] * 1000000  # Convert to µg
            
            print(f"   Energy per Inference: {energy_per_inference:.2f} µWh")
            print(f"   CO₂ per Inference: {co2_per_inference:.2f} µg")
        else:
            print(f"   Energy tracking not available")
        
        print(f"\n🖥️ System Resources:")
        print(f"   Avg CPU Usage: {system['avg_cpu_percent']:.1f}%")
        print(f"   Avg Memory Usage: {system['avg_memory_percent']:.1f}%")
        
        if system['avg_temperature_celsius']:
            print(f"   Avg Temperature: {system['avg_temperature_celsius']:.1f}°C")
            print(f"   Max Temperature: {system['max_temperature_celsius']:.1f}°C")
            
            if system['thermal_throttling_detected']:
                print(f"   ⚠️ Thermal throttling detected!")
            else:
                print(f"   ✅ No thermal throttling")
        
        print(f"\n🏆 Edge Optimization Score:")
        # Calculate optimization score based on multiple factors
        score_factors = []
        
        # Inference speed (lower is better, target < 100ms)
        speed_score = max(0, 100 - benchmark['avg_inference_time_ms']) / 100 * 25
        score_factors.append(('Speed', speed_score))
        
        # Resource efficiency (lower CPU/memory usage is better)
        resource_score = max(0, 100 - (system['avg_cpu_percent'] + system['avg_memory_percent']) / 2) / 100 * 25
        score_factors.append(('Resource Efficiency', resource_score))
        
        # Thermal management (staying under limit is good)
        if system['avg_temperature_celsius']:
            thermal_score = max(0, 100 - (system['max_temperature_celsius'] / self.power_profile['thermal_limit_celsius'] * 100)) / 100 * 25
        else:
            thermal_score = 25  # Assume good if no temperature data
        score_factors.append(('Thermal Management', thermal_score))
        
        # Energy efficiency (if available)
        if benchmark['energy_consumed_kwh']:
            energy_per_inference_wh = benchmark['energy_consumed_kwh'] / benchmark['total_inferences'] * 1000
            # Target < 0.001 Wh per inference for edge devices
            energy_score = max(0, min(25, (0.001 - energy_per_inference_wh) / 0.001 * 25))
        else:
            energy_score = 12.5  # Neutral score if no energy data
        score_factors.append(('Energy Efficiency', energy_score))
        
        total_score = sum(score for _, score in score_factors)
        
        print(f"   Overall Score: {total_score:.1f}/100")
        for factor, score in score_factors:
            print(f"   - {factor}: {score:.1f}/25")
        
        if total_score >= 80:
            print(f"   🏆 Excellent edge optimization!")
        elif total_score >= 60:
            print(f"   ✅ Good edge performance")
        elif total_score >= 40:
            print(f"   ⚠️ Moderate edge performance")
        else:
            print(f"   ❌ Needs optimization for edge deployment")

def main():
    """Main function for edge runner"""
    print("🌍 Terravex Edge AI Runner")
    print("=" * 40)
    
    # Detect device type
    device_type = 'raspberry_pi'  # Default
    
    # Try to detect device type automatically
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read().lower()
            if 'raspberry pi' in cpuinfo:
                device_type = 'raspberry_pi'
            elif 'jetson' in cpuinfo:
                device_type = 'jetson_nano'
    except:
        # Check memory to guess device type
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < 2:
            device_type = 'raspberry_pi'
        elif memory_gb < 8:
            device_type = 'jetson_nano'
        elif memory_gb < 16:
            device_type = 'laptop'
        else:
            device_type = 'desktop'
    
    print(f"🔍 Detected device type: {device_type}")
    
    # Initialize edge runner
    edge_runner = EdgeGreenAI(device_type)
    
    # Create a simple test model
    import torch.nn as nn
    
    class SimpleTestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(32, 10)
            
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    # Create test model and data
    model = SimpleTestModel()
    test_data = [torch.randn(3, 32, 32) for _ in range(50)]  # 50 test samples
    
    print(f"🧪 Running edge benchmark with {len(test_data)} samples...")
    
    # Run benchmark
    results = edge_runner.run_edge_benchmark(model, test_data, num_samples=50)
    
    print(f"\n✅ Edge benchmark completed!")
    print(f"📄 Results saved to: edge_benchmark_{device_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

if __name__ == "__main__":
    main()