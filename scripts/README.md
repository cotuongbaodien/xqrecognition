## Generate dataset

input: board empty image `scripts/ban_co.png` and pieces images for 14 classes `scripts/pieces`
process: generate dataset by random placing pieces on the board about 5000 images for train and 1000 images for valid.
output: `scripts/dataset`

```bash
python generate_chess_dataset.py
```

## Split dataset

input: `scripts/dataset` copy or download images from online resources and put them in `scripts/dataset/images` and labels in `scripts/dataset/labels`
process: split dataset into train and valid by naming convention.
output: `scripts/dataset/train` and `scripts/dataset/valid`

```bash
python split_dataset.py
```