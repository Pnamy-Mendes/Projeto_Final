import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def check_install(package, version=None):
    try:
        if version:
            pkg = __import__(package.split('==')[0])
            if pkg.__version__ == version:
                print(f"{package} is already installed.")
            else:
                print(f"{package} is installed but the version is not correct. Installing the correct version.")
                install(package)
        else:
            __import__(package)
            print(f"{package} is already installed.")
    except ImportError:
        print(f"{package} is not installed. Installing now.")
        install(package)

def install_requirements_from_file(file_path):
    with open(file_path, 'r') as file:
        requirements = file.readlines()
    
    for requirement in requirements:
        requirement = requirement.strip()
        if "==" in requirement:
            pkg, ver = requirement.split("==")
            check_install(pkg, ver)
        else:
            check_install(requirement)

if __name__ == "__main__":
    install_requirements_from_file('requirements.txt')
