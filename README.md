# Deep Head Pose V2 #

Welcome to the deep-head-pose-v2 repo.

The work is described in the pdf file in the root of the project.

We also share our trained models that can be used to validate results from the paper:

[Master18](https://drive.google.com/file/d/14YcF9qOIgG96VcKPFnm50V7Lywp__iXN/view?usp=sharing)

[Master50](https://drive.google.com/file/d/10sNFYcgL5uyWWzBBqdYnx5Ag5JlKTKcz/view?usp=sharing)

[Master101](https://drive.google.com/file/d/1abcQThhY5wVZ150IRjO3FnjFcYZrk0wu/view?usp=sharing)

To perform inference on one of the models, on the AFLW2000 dataset use:
```bash
python test.py --data_dir path/to/data --filename_list path/to/filenames --dataset "AFLW2000" --model 'ResNet50' --snapshot path/to/master50.pkl
```

To train a model from scratch, do
```bash
python train_hopenet.py --data_dir path/to/data --filename_list path/to/filenames --num_epochs 35 --model 'ResNet50' ...
```
