import subprocess

# List of explainability scripts
scripts = [
    "explainability_lime.py",
    "explainability_gradcam.py",
    "explainability_gradcam_all.py",
    "explainability_occlusion_sensitivity.py",
    "explainability_activations.py"
]

print("=== Running all explainability scripts ===\n")
for script in scripts:
    print(f"Running {script}...")
    subprocess.run(["python", script])
    print(f"Finished {script}\n")

print("All explainability methods have been executed!")
