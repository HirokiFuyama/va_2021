import sys
import glob
import soundfile as sf


def read_wav(file_dir_path: str) -> list:
    """ read wav files

    Args:
        file_dir_path (str): path of wav files directory

    Returns:
        raw_data (list): wav data
        file_path (list) : path of wav files

    """
    file_path = glob.glob(file_dir_path)
    file_path.sort()
    raw_data = [sf.read(file_path[i])[0] for i in range(len(file_path))]
    if file_path == []:
        print('FileNotFoundError: No such file or directory: ', file=sys.stderr)
        sys.exit(1)
    return raw_data, file_path
