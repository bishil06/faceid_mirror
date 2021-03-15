import os

def get_ext(file_name):
    name, ext = os.path.splitext(file_name)
    return ext

def get_file_list(path, ext_list=['.png', '.jpg']):
    dir = os.listdir(path)
    files = [file_path for file_path in dir if get_ext(file_path) in ext_list]
    return files

# flist = get_file_list('.', ['.png', '.jpg'])
# print(flist)