import platform
import psutil
import subprocess
import sys
from datetime import datetime

def get_cpu_info():
    return {
        "Processor": platform.processor(),
        "Physical cores": psutil.cpu_count(logical=False),
        "Total cores": psutil.cpu_count(logical=True),
        "Max Frequency (MHz)": psutil.cpu_freq().max if psutil.cpu_freq() else "N/A"
    }

def get_ram_info():
    svmem = psutil.virtual_memory()
    return {
        "Total RAM (GB)": round(svmem.total / (1024**3), 2),
        "Available RAM (GB)": round(svmem.available / (1024**3), 2)
    }

def get_gpu_info():
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            encoding='utf-8'
        )
        gpus = result.strip().split("\n")
        return gpus
    except Exception:
        return ["No NVIDIA GPU found or nvidia-smi not available"]

def get_installed_packages():
    try:
        import pkg_resources
        packages = sorted(["%s==%s" % (i.key, i.version) for i in pkg_resources.working_set])
        return packages
    except Exception:
        return ["Could not retrieve packages"]

def get_python_info():
    return {
        "Python Version": sys.version,
        "Executable": sys.executable
    }

def get_os_info():
    return {
        "System": platform.system(),
        "Node Name": platform.node(),
        "Release": platform.release(),
        "Version": platform.version(),
        "Machine": platform.machine()
    }

def write_to_file(filename="system_info.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("===== SYSTEM INFORMATION =====\n")
        f.write(f"Generated at: {datetime.now()}\n\n")

        f.write("---- OS INFO ----\n")
        for k, v in get_os_info().items():
            f.write(f"{k}: {v}\n")

        f.write("\n---- CPU INFO ----\n")
        for k, v in get_cpu_info().items():
            f.write(f"{k}: {v}\n")

        f.write("\n---- RAM INFO ----\n")
        for k, v in get_ram_info().items():
            f.write(f"{k}: {v}\n")

        f.write("\n---- GPU INFO ----\n")
        for gpu in get_gpu_info():
            f.write(f"{gpu}\n")

        f.write("\n---- PYTHON INFO ----\n")
        for k, v in get_python_info().items():
            f.write(f"{k}: {v}\n")

        f.write("\n---- INSTALLED PACKAGES ----\n")
        for pkg in get_installed_packages():
            f.write(f"{pkg}\n")

if __name__ == "__main__":
    write_to_file()
    print("System information saved to system_info.txt")