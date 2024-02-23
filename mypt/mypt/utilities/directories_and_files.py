"""
This scripts contains functionalities to manipulate files and directories
"""
import os
import zipfile
import shutil
import re
import numpy as np

from typing import Union, Optional, List
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split

HOME = os.path.dirname(os.path.realpath(__file__))


def abs_path(path: Union[str, Path]) -> Path:
    return Path(path) if os.path.isabs(path) else Path(os.path.join(HOME, path))


DEFAULT_ERROR_MESSAGE = 'MAKE SURE THE passed path satisfies the condition passed with it'


def process_path(save_path: Union[str, Path, None],
                      dir_ok: bool = True,
                      file_ok: bool = True,
                      condition: callable = None,
                      error_message: str = DEFAULT_ERROR_MESSAGE) -> Union[str, Path, None]:
    if save_path is not None:
        if not os.path.isfile(save_path) and dir_ok and not os.path.exists(save_path):
            os.makedirs(save_path)
            
        # first make the save_path absolute
        save_path = abs_path(save_path)
        
        assert not \
            ((not file_ok and os.path.isfile(save_path)) or
             (not dir_ok and os.path.isdir(save_path))), \
            f'MAKE SURE NOT TO PASS A {"directory" if not dir_ok else "file"}'

        assert condition is None or condition(save_path), error_message


    return save_path


def default_file_name(hour_ok: bool = True,
                      minute_ok: bool = True):
    # Get timestamp of current date (all experiments on certain day live in same folder)
    current_time = datetime.now()
    current_hour = current_time.hour
    current_minute = current_time.minute
    timestamp = datetime.now().strftime("%Y-%m-%d")  # returns current date in YYYY-MM-DD format
    # now it is much more detailed: better tracking
    timestamp += f"-{(current_hour if hour_ok else '')}-{current_minute if minute_ok else ''}"

    # make sure to remove any '-' left at the end
    timestamp = re.sub(r'-+$', '', timestamp)
    return timestamp


def squeeze_directory(directory_path: Union[str, Path]) -> None:
    # Given a directory with only one subdirectory, this function moves all the content of
    # subdirectory to the parent directory

    # first convert to abs
    path = abs_path(directory_path)

    if not os.path.isdir(path):
        return

    files = os.listdir(path)
    if len(files) == 1 and os.path.isdir(os.path.join(path, files[0])):
        subdir_path = os.path.join(path, files[0])
        # copy all the files in the subdirectory to the parent one
        for file_name in os.listdir(subdir_path):
            shutil.move(src=os.path.join(subdir_path, file_name), dst=path)
        # done forget to delete the subdirectory
        os.rmdir(subdir_path)


def copy_directories(src_dir: str,
                     des_dir: str,
                     copy: bool = True,
                     filter_directories: callable = None) -> None:
    # convert the src_dir and des_dir to absolute paths
    src_dir, des_dir = abs_path(src_dir), abs_path(des_dir)

    # create the directories if needed
    os.makedirs(src_dir, exist_ok=True)
    os.makedirs(des_dir, exist_ok=True)

    assert os.path.isdir(src_dir) and os.path.isdir(des_dir), "BOTH ARGUMENTS MUST BE DIRECTORIES"

    if filter_directories is None:
        def filter_directories(_):
            return True

    # iterate through each file in the src_dir
    for file_name in os.listdir(src_dir):
        file_path = os.path.join(src_dir, file_name)
        # move / copy
        if filter_directories(file_name):
            if copy:
                if os.path.isdir(file_path):
                    shutil.copytree(file_path, os.path.join(des_dir, file_name))
                else:
                    shutil.copy(file_path, des_dir)
            else:
                shutil.move(file_path, des_dir)

    # remove the source directory if it is currently empty
    if len(os.listdir(src_dir)) == 0:
        shutil.rmtree(src_dir)


def unzip_data_file(data_zip_path: Union[Path, str],
                    unzip_directory: Optional[Union[Path, str]] = None,
                    remove_inner_zip_files: bool = True) -> Union[Path, str]:
    data_zip_path = abs_path(data_zip_path)

    assert os.path.exists(data_zip_path), "MAKE SURE THE DATA'S PATH IS SET CORRECTLY!!"

    if unzip_directory is None:
        unzip_directory = Path(data_zip_path).parent

    unzipped_dir = os.path.join(unzip_directory, os.path.basename(os.path.splitext(data_zip_path)[0]))
    os.makedirs(unzipped_dir, exist_ok=True)

    # let's first unzip the file
    with zipfile.ZipFile(data_zip_path, 'r') as zip_ref:
        # extract the data to the unzipped_dir
        zip_ref.extractall(unzipped_dir)

    # unzip any files inside the subdirectory
    for file_name in os.listdir(unzipped_dir):
        file_path = os.path.join(unzipped_dir, file_name)
        if zipfile.is_zipfile(file_path):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                # extract the data to current directory
                zip_ref.extractall(unzipped_dir)

            # remove the zip files if the flag is set to True
            if remove_inner_zip_files:
                os.remove(file_path)

    # squeeze all the directories
    for file_name in os.listdir(unzipped_dir):
        squeeze_directory(os.path.join(unzipped_dir, file_name))

    return unzipped_dir


def dataset_portion(directory_with_classes: Union[str, Path], 
                    destination_directory: Union[str, Path] = None,
                    portion: float = 0.1, 
                    copy: bool = False, 
                    seed: int = 69) -> Union[str, Path]:
    
    # make sure the portion is a float between '0' and '1'
    if not (isinstance(portion, float) and 1 >= portion > 0):
        raise ValueError(f"The portion of the dataset is expected to be a number from '0' to '1'.Found: {portion}")

    # the first step is to process the passed path
    def all_inner_files_directories(path):
        return all([
            os.path.isdir(os.path.join(path, d)) for d in os.listdir(path)
        ])

    src = process_path(directory_with_classes,
                            dir_ok=True,
                            file_ok=False,
                            condition=lambda path: all_inner_files_directories(path),
                            error_message='ALL FILES IN THE PASSED DIRECTORIES MUST BE DIRECTORIES')

    # set the default location of the destination directory
    des = os.path.join(Path(directory_with_classes).parent, f'{os.path.basename(src)}_{portion}') \
        if destination_directory is None else destination_directory

    # process the path
    des = process_path(des, file_ok=False, dir_ok=True)    
    # for each dri

    for src_dir in os.listdir(src):
        des_dir = process_path(os.path.join(des, src_dir), file_ok=False)
        src_dir = os.path.join(src, src_dir)

        src_dir_files = np.asarray(os.listdir(src_dir))

        # split the data 
        _, files_move = train_test_split(src_dir_files, test_size=portion, random_state=seed)
        # define the criterion 
        files_move = set(files_move.tolist())

        def filter_callable(file_name):
            return file_name in files_move
        
        copy_directories(src_dir, des_dir, copy=copy, filter_directories=filter_callable)
        
    return Path(des)


_IMAGE_EXTENSIONS = ['.png', '.jpeg', '.jpg']


def image_directory(path: Union[Path, str], image_extensions = None) -> bool:
    if image_extensions is None:
        image_extensions = _IMAGE_EXTENSIONS

    for file in os.listdir(path):
        _, ext = os.path.splitext(file)
        if ext not in image_extensions:
            return False
    return True

def image_dataset_directory(path: Union[Path, str], 
                            image_extensions: List[str] = None) -> bool:
    if image_extensions is None:
        image_extensions = _IMAGE_EXTENSIONS
    
    # the path should point to a directory
    if not os.path.isdir(path):
        return False
    
    for p in os.listdir(path):
        folder_path = os.path.join(path, p)
        # first check it is a directory, return False otherwise
        if not os.path.isdir(folder_path):
            return False
        # check if the inner directory contains only images
        if not image_directory(folder_path):
            return False

    return True

def image_dataset_directory_error(path: Union[Path, str]) -> str: 
    return f"Please make sure the path: {path} contains only directories for classes and each class directory contains image files."

def clear_directory(directory: Union[str, Path], 
                 condition: callable):
    """This function removes any file (or directory) that satisfies a given condition in a given directory

    Args:
        directory (Union[str, Path]): _description_
    """
    # process the path
    directory = process_path(directory, dir_ok=True, file_ok=False)
    # create a walk object
    walk = os.walk(directory)

    for r, dir, files in walk: 
        # first iterate through the directories in 
        for d in dir: 
            path = os.path.join(r, d)
            if condition(path):
                shutil.rmtree(path)

        # iterate through files
        for f in files:
            p = os.path.join(r, f)
            if condition(p):
                os.remove(p)

