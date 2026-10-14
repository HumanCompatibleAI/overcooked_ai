import sys
import os

print("--- Python Environment Check ---")
print(f"Python Version: {sys.version.split()[0]}")
print(f"Executable Path: {sys.executable}")
print(f"Working Directory: {os.getcwd()}")
print("----------------------------------")