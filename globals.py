from os import path

LINEAR_DEFLECTION = 0.08
ANGULAR_DEFLECTION = 0.2617  # [rad] = 15°
ENABLE_MULTICORE = False  # Only works well with Linux at the moment :( Api not thread save?! - Might crash!
RUNNING_OS = 'w'
TMP_FOLDER_PATH = path.join(path.dirname(path.realpath(__file__)), "tmp_files")
ROOT_FOLDER_PATH = path.dirname(path.realpath(__file__))
