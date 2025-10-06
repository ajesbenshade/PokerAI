import psutil
import time
import subprocess
import os
import torch

# Get training PID
try:
    result = subprocess.run(['pgrep', '-f', 'python train.py'], capture_output=True, text=True)
    pid = int(result.stdout.strip())
    print(f'Monitoring PID {pid}')
except:
    print('Could not find training PID')
    exit(1)

ram_threshold = 55 * 1024**3  # 55GB
vram_threshold = 14 * 1024**3  # 14GB

for i in range(120):  # Monitor for 20 minutes (120 * 10s)
    try:
        proc = psutil.Process(pid)
        ram_usage = proc.memory_info().rss
        print(f'Iteration {i}: Process RAM {ram_usage / 1024**3:.1f}GB')

        # Check system RAM
        mem = psutil.virtual_memory()
        print(f'System RAM {mem.used / 1024**3:.1f}GB / {mem.total / 1024**3:.1f}GB')

        # Check VRAM - support both CUDA and ROCm
        vram_usage = 0
        if torch.cuda.is_available():
            vram_usage = torch.cuda.memory_allocated() / 1024**3
        elif hasattr(torch, 'hip') and torch.hip.is_available():
            # For ROCm/HIP
            vram_usage = torch.hip.memory_allocated() / 1024**3
        elif torch.backends.mps.is_available():
            # For Apple Silicon
            vram_usage = torch.mps.current_allocated_memory() / 1024**3
        
        if vram_usage > 0:
            print(f'VRAM {vram_usage:.1f}GB')
        else:
            # Fallback: try to get GPU memory from system
            try:
                import subprocess
                result = subprocess.run(['rocm-smi', '--showmeminfo', 'vram', '--csv'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for line in lines[1:]:  # Skip header
                        parts = line.split(',')
                        if len(parts) >= 3 and 'GPU' in parts[0]:
                            used_mb = int(parts[2])  # Used memory in MB
                            vram_usage = used_mb / 1024  # Convert to GB
                            print(f'VRAM {vram_usage:.1f}GB (ROCm)')
                            break
            except:
                print('VRAM 0.0GB')

        if ram_usage > ram_threshold or mem.used > 55 * 1024**3 or vram_usage > vram_threshold:
            print('Threshold exceeded, adjusting config...')
            # Adjust config
            with open('config.py', 'r') as f:
                content = f.read()
            # Reduce NUM_SIMULATIONS and MAX_WORKERS
            content = content.replace('NUM_SIMULATIONS = 8', 'NUM_SIMULATIONS = 6')
            content = content.replace('MAX_WORKERS = 6', 'MAX_WORKERS = 4')
            with open('config.py', 'w') as f:
                f.write(content)
            # Kill and restart
            os.kill(pid, 9)
            print('Killed old process, restarting...')
            subprocess.run(['python', 'train.py'])
            break
    except psutil.NoSuchProcess:
        print('Training process ended')
        break
    except Exception as e:
        print(f'Monitoring error: {e}')
    time.sleep(10)
print('Monitoring complete')
