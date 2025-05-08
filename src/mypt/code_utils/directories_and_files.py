"""
This scripts contains functionalities to manipulate files and directories
"""
import os, zipfile, shutil, re, random
import numpy as np

from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from typing import Union, Optional, List, Dict, Callable, Tuple

HOME = os.getcwd()


def abs_path(path: Union[str, Path]) -> Path:
    return Path(path) if os.path.isabs(path) else Path(os.path.join(HOME, path))


DEFAULT_ERROR_MESSAGE = 'MAKE SURE THE passed path satisfies the condition passed with it'
IMAGE_EXTENSIONS = ['.png', '.jpeg', '.jpg']


def process_path(save_path: Union[str, Path, None],
                      dir_ok: bool = True,
                      file_ok: bool = True,
                      must_exist: bool = False,
                      condition: Callable = None,
                      error_message: str = DEFAULT_ERROR_MESSAGE) -> Union[str, Path, None]:

    if save_path is None:
        return None

    # first make the save_path absolute
    save_path = abs_path(save_path)

    if not os.path.exists(save_path):
        if must_exist:
            raise ValueError(f"Passing a non-existing directory while setting the 'must_exist' argument to True")

        if dir_ok and not file_ok:
            os.makedirs(save_path)
    
        else:
            raise ValueError(f"when passing a non-existing file, the parameters dir_ok and file_ok must be set to True and False respectively")

    assert not \
        ((not file_ok and os.path.isfile(save_path)) or
            (not dir_ok and os.path.isdir(save_path))), \
        f'MAKE SURE NOT TO PASS A {"directory" if not dir_ok else "file"}'

    if condition is not None:
        condition_output = condition(save_path)
        # check the type of the condition output
        if not isinstance(condition_output, (tuple, bool)):
            raise TypeError(f"The condition function must output either a tuple[bool, str] where 'str' can represent the error, or a 'bool' value. Found: {condition_output}")

        if isinstance(condition_output, Tuple):
            res, possible_error = condition_output
            # make sure the types are correct (since we cannot use generic types with the 'isinstance' function) 
            if not (isinstance(res, bool) and isinstance(possible_error, str)):
                raise TypeError(f"if the condition function outputs a tuple. The first element must be a bool and the second a str. found: {condition_output}")

            if not res:
                raise ValueError(f"general error message: {error_message}\n specific error message: {possible_error}")

        # at this point, we know that 'condition_output' is a boolean variable 
        # only consider the case of condition_output = False
        elif not condition_output:
            raise ValueError(error_message)
                
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
        for file_name in sorted(os.listdir(subdir_path)):
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
    for file_name in tqdm(sorted(os.listdir(src_dir)), desc='copying files'):
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
    for file_name in sorted(os.listdir(unzipped_dir)):
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


def classification_ds_partition(directory_with_classes: Union[str, Path], 
                        destination_directory: Union[str, Path] = None,
                        portion: Union[int, float] = 0.1, 
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

    for src_dir in sorted(os.listdir(src)):
        des_dir = process_path(os.path.join(des, src_dir), file_ok=False)
        src_dir = os.path.join(src, src_dir)

        if portion == 1.0:
            copy_directories(src_dir, des_dir, copy=copy)
            continue

        src_dir_files = np.asarray(sorted(os.listdir(src_dir)))
        # split the data 
        _, files_move = train_test_split(src_dir_files, test_size=portion, random_state=seed)
        # define the criterion 
        files_move = set(files_move.tolist())

        def filter_callable(file_name):
            return file_name in files_move
        
        copy_directories(src_dir, des_dir, copy=copy, filter_directories=filter_callable)
        
    return Path(des)


def directory_partition(src_dir: Union[str, Path], 
                        des_dir: Union[str, Path],
                        portion: Union[int, float] = 0.1, 
                        copy: bool = False, 
                        seed: int = 69) -> Union[str, Path]:

    # make sure the portion is a float between '0' and '1'
    if not (isinstance(portion, float) and 1 >= portion > 0):
        raise ValueError(f"The portion of the dataset is expected to be a number from '0' to '1'.Found: {portion}")

    src_dir = process_path(src_dir, dir_ok=True, file_ok=False)
    # process the path
    des_dir = process_path(des_dir, file_ok=False, dir_ok=True)    

    # sorting the files ensures the reproducibility of the function across different systems (since os.listdir is not uniform across different platforms)
    src_dir_files = np.asarray(sorted(os.listdir(src_dir)))

    # split the data 
    _, files_move = train_test_split(src_dir_files, test_size=portion, random_state=seed)
    # define the criterion 
    files_move = set(files_move.tolist())
    
    copy_directories(src_dir, des_dir, copy=copy, filter_directories=lambda f: f in files_move)
        
    return Path(des_dir)


def image_directory(path: Union[Path, str], image_extensions = None) -> bool:
    if image_extensions is None:
        image_extensions = IMAGE_EXTENSIONS

    for file in os.listdir(path):
        _, ext = os.path.splitext(file)
        if ext not in image_extensions:
            return False
    return True


def image_dataset_directory(path: Union[Path, str], 
                            image_extensions: List[str] = None) -> bool:
    if image_extensions is None:
        image_extensions = IMAGE_EXTENSIONS
    
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


def dir_contains_only_types(directory:Union[str, Path], valid_extensions: List[str]) -> Tuple[bool, str] | bool:
    # make sure to convert the extensions to lower case
    valid_extensions = [e.lower() for e in valid_extensions]
    directory = process_path(directory, dir_ok=True, file_ok=False)
    for r, _, files in os.walk(directory):
        for f in files:
            file_path = os.path.join(r, f)
            if os.path.splitext(file_path)[-1].lower() not in valid_extensions:
                return False, f"the file {os.path.basename(file_path)} has a disallowed extension."
    
    return True


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


def compare_directories(dir1: Union[str, Path], dir2: Union[str, Path]) -> bool:
    dir1 = process_path(dir1, dir_ok=True, file_ok=False)
    dir2 = process_path(dir2, dir_ok=True, file_ok=False)

    # no point in going further if the number of items inside isn't the same    
    if len(os.listdir(dir1)) != len(os.listdir(dir2)):
        return False

    files1, files2 = sorted(os.listdir(dir1)), sorted(os.listdir(dir2))
    for f1, f2 in zip(files1, files2):
        if f1 != f2:
            return False

        one_step_lower = True
        if os.path.isdir(os.path.join(dir1, f1)):
            one_step_lower = compare_directories(dir1=os.path.join(dir1, f1), dir2=os.path.join(dir2, f2))
        
        if not one_step_lower:
            return False
    
    return True


# the following functions are created to split a given directory into 'n' sub directories.
def _assign_files(dir: Union[str, Path], num_splits: int, seed: int) -> Dict:
    """
    this function assigns a file name to each split and returns the mapping as a dictionary object
    """
    # set the seed
    random.seed(seed)
    np.random.seed(seed)

    if  num_splits < 2:
        raise ValueError(f"the number of splits is expected to be at least 2. Found: {num_splits}")
    

    dir = process_path(dir, file_ok=False, condition=lambda p: len(os.listdir(p)) > 0, error_message="the directory cannot be empty !!!")
    # a dictionary to mape the current number of files in each split
    split_file_count = {i: 0 for i in range(1, num_splits + 1)}
    # the split: file name mapping
    split_file_map = {i : [] for i in range(1, num_splits + 1)}
    
    avg_file_per_split = len(os.listdir(dir)) / num_splits

    # this variable will store the splits for which we can assign a file name (whose count is yet to exceed_split_file_per_split)
    valid_splits = set(range(1, num_splits + 1))

    random_permutation = np.random.permutation(len(os.listdir(dir)))
    files = [os.listdir(dir)[i] for i in random_permutation]
    
    for file_name in files:
        full_splits = [i for i in valid_splits if split_file_count[i] >= avg_file_per_split]
        valid_splits.difference_update(set(full_splits))

        # choose the split with the least number of files
        split = min(valid_splits, key=split_file_count.get)
        # add the file to the split
        split_file_map[split].append(file_name)
        # make sure to keep up the count
        split_file_count[split] += 1
    
    # quick checks for the correctness of the code 
    for s in range(1, num_splits + 1):
        if split_file_count[s] != len(split_file_map[s]):
            raise ValueError(f"The count for a ginve split does not match the actual number of assigned files")
    
    # make sure the splits are disjoint
    for s1 in range(1, num_splits + 1):
        for s2 in range(s1 + 1, num_splits + 1):
            if len(set(split_file_map[s1]).intersection(split_file_map[s2])) != 0:
                raise ValueError(f"a common file was found between split {s1} and split {s2}")
    
    # make sure the file counts are close to each other
    max_count, min_count = max(list(split_file_count.values())), min(list(split_file_count.values()))

    # supposedly each 
    if (max_count - min_count)  > num_splits:
        raise ValueError(f"the splits are too imbalanced")

    return split_file_map


def split_dir_disjoint(dir: Union[str, Path], 
                       num_splits: int,
                       splits_dirs: List[Union[str, Path]] = None,
                       seed: int = 69, 
                       copy: bool = True
                       ):
    # let's assign each split its list of file names
    split_file_map = _assign_files(dir=dir, num_splits=num_splits, seed=seed)

    if splits_dirs is None:
        splits_dirs [os.path.join()]

    if len(splits_dirs) != num_splits:
        raise ValueError("each split should be assigned a parent directory")

    for d in splits_dirs:
        if not os.path.exists(d):
            os.makedirs(d)

    # the next step is to simply is to either move or copy the files from the original folder to the splits
    for i, file_names in split_file_map.items():
        for fn in file_names:
            if copy:
                shutil.copyfile(src=os.path.join(dir, fn), dst=os.path.join(splits_dirs[i - 1], fn))
            else:
                shutil.move(src=os.path.join(dir, fn), dst=os.path.join(splits_dirs[i - 1], fn))
    
    if not copy and len(os.listdir(dir)) == 0:
        shutil.rmtree(dir)
