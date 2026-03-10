import os
import shutil
import src.sincfold as sincfold

command = "train"
target_file = "sample/train.csv"
out_path = "output_path/"
# if directory exists, delete it
if os.path.isdir(out_path):
    shutil.rmtree(out_path)

config= {"device": "cpu", "batch_size": 1, "max_epochs": 1, 
         "valid_split": 0.1, "max_len": 512, "verbose": True, "cache_path": "cache/"}

if config["cache_path"] is not None:
        shutil.rmtree(config["cache_path"], ignore_errors=True)
        os.makedirs(config["cache_path"])

if command == "train": 
    sincfold.train(target_file, config, out_path, valid_file=None, nworkers=2)


