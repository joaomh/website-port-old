from shutil import copy
import os 
import argparse
import sys

def copy_img(input_img):
    """
    Copy an image and rename to all format used in the blog.
    The original image must have a good resolution.

    Parameters
    ----------
    input_img : str
        Name of the img, must be in the assets/img/posts folder.

    Returns
    -------
    None.

    """
    src_path = 'assets/img/posts/{}.jpg'.format(input_img)
    os.path.join(os.getcwd(), src_path)
    list_img = ['lg','md','placehold','sm','thumb','thumb@2x','xs']
    for i in list_img:
        dest_path = 'assets/img/posts/{}_{}.jpg'.format(input_img,i)
        copy(os.path.join(os.getcwd(), src_path),(os.path.join(os.getcwd(), dest_path)))
    print('File copied and renamed successfully!')

if __name__ == '__main__':
    copy_img(sys.argv[1])