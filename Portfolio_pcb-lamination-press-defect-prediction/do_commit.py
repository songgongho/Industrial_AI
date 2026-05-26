#!/usr/bin/env python
import subprocess
import os

os.chdir(r'E:\2026-1학기\Industrial_AI\Portfolio_pcb-lamination-press-defect-prediction')

# Commit
result1 = subprocess.run(['git', 'commit', '-m', 'Update: Add literature review and quality dashboard'],
                        capture_output=True, text=True)
print("Commit output:")
print(result1.stdout)
if result1.stderr:
    print("Errors:", result1.stderr)
print("Return code:", result1.returncode)

# Status
result2 = subprocess.run(['git', 'status', '--short'],
                        capture_output=True, text=True)
print("\nCurrent status:")
print(result2.stdout[:500])

# Push
print("\n--- Pushing to GitHub ---")
result3 = subprocess.run(['git', 'push', 'origin', 'main'],
                        capture_output=True, text=True)
print(result3.stdout)
if result3.stderr:
    print(result3.stderr)

