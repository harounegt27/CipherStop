import pandas as pd
from pandas.api.types import is_integer_dtype
import time

current_time_stamp = time.time()


def check_files(paths,i,j):
    for path in paths:
        data = pd.read_csv(path)

        Unix_sec = data.iloc[:, i]
        Unix_nsec = data.iloc[:, j]
        Unix_timestamp = Unix_sec + (Unix_nsec / 1000000000)

        if is_integer_dtype(Unix_sec) and is_integer_dtype(Unix_nsec):
            pass
        else:
            print(f"{path}: columns contains not integer values")
        if Unix_sec.isnull().any() or Unix_nsec.isnull().any():
            print(f"{path} columns contains null values")

        if (Unix_timestamp >= current_time_stamp).any():
            print(f"{path} Unix_timestamp contains timestamps in the future")

        if (Unix_timestamp < 0).any():
            print(f"{path} Unix_timestamp contains negative values")



def check_lba(paths, i):
    for path in paths:
        data = pd.read_csv(path)
        lba = data.iloc[:, i]

        if is_integer_dtype(lba):
            pass
        else:
            print(f"{path}, columns contains non integer values")
        if lba.isnull().any():
            print(f"{path}, columns contains null values")



def check_size(paths, i):
    for path in paths:
        data = pd.read_csv(path)
        size = data.iloc[:, i]

        if is_integer_dtype(size):
            pass
        else:
            print(f"{path}, column contains non-integer values")

        if size.isnull().any():
            print(f"{path}, column contains null values")
            


def check_entropy(paths, i):
    for path in paths:
        data = pd.read_csv(path)
        entropy = pd.to_numeric(data.iloc[:, i], errors="coerce")

        if entropy.isnull().any():
            print(f"{path}: entropy contains null or non-numeric values")

        invalid_entropy = (entropy < 0) | (entropy > 1)

        if invalid_entropy.any():
            print(f"{path}: entropy contains values outside [0, 1]")



def check_gpa(paths,i):
    for path in paths :
        data = pd.read_csv(path)
        gpa = data.iloc[:, i]
        
        if gpa.isnull().any():
            print("GPA column contains null values")
        
        if not is_integer_dtype(gpa):
            print("GPA column contains non-integer values")



def check_type(paths, i, allowed=(1, 2, 3, 4)):
    for path in paths:
        data = pd.read_csv(path)
        type_col = data.iloc[:, i]

        if is_integer_dtype(type_col):
            pass
        else:
            print(f"{path}, type column contains non-integer values")

        if type_col.isnull().any():
            print(f"{path}, type column contains null values")

        if not type_col.isin(allowed).all():
            print(f"{path}, type column contains invalid values")


