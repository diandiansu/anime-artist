# anime-artist
1. Download github repo: https://github.com/eriklindernoren/PyTorch-GAN
2. Separate Dataset into Domain A and B
3. Tried cyclegan with our own datasets

```
python3 cyclegan.py --dataset_name=draw2cartoon --batch_size=8 --img_height=64 --img_width=64 --n_residual_blocks=3

```
```
python3 discogan.py --n_epochs=50 --dataset_name=draw2cartoon --img_height=64 --img_width=64 --batch_size=8 
```
```
python3 discogan.py --n_epochs=100 --dataset_name=APdrawing2Anime --img_height=64 --img_width=64 --batch_size=8 
```
```
python3 cyclegan.py --n_epochs=100 --dataset_name=APdrawing2Anime --img_height=64 --img_width=64 --batch_size=8 
```
