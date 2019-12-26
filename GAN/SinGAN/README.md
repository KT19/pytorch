# Implementation of SinGAN: Learning a Generative Model from a Single Natural Image

## Usage
***
python3 train.py --device device --data_dir path_to_data_dir --img_name img_folder --ref_img img.jpg
***

### option(necessary)
--data_dir: is a path to image folders(there are many image folders)

--img_name: is a path to image folder(for one image folder)

--ref_img: is a path to reference image in image folder(this argument determine one image in training)

## Files
1. train.py (for training)
2. eval.py (for evaluation)
3. modules.py (define each modules)
4. data_loader.py (for data loader)
5. configure.py (training option)

If you want to check arguments, please check configure.py.
All arguments or training option is defined in this file.
