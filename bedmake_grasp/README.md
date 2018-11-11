# Benchmarking Grasp Network in PyTorch

Here's how to (roughly) reproduce the results I got from earlier with grasping.
This is detection so one needs to fix the target labels during data
augmentation. Use `custom_transforms.py` for my transformations.

Run `prepare_data.py` to prepare the data. Then run `grasp.py` to train. 
