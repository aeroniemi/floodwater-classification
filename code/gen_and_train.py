import argparse
import subprocess
parser = argparse.ArgumentParser(description='Generate feature space and then run a model')
parser.add_argument('feature_space', metavar='f', type=str)
parser.add_argument('model', type=str)

args = parser.parse_args()

if __name__ == "__main__":
    print(f"Build and run with {args.feature_space} and {args.model}")
    subprocess.run(["python", "generate_dataset.py", args.feature_space])
    subprocess.run(["python", "train_models.py", args.feature_space, args.model])