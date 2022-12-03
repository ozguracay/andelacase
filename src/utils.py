import zipfile


def unzip_file(file_path: str, dest_dir: str):
    """
    Unzip given file
    Arguments:
        file_path: path of zipped file
        dest_dir: dir of unzipped file
    """
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(dest_dir)
