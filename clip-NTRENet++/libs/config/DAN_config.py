
import os


from easydict import EasyDict
OPTION = EasyDict()
OPTION.root_path = os.path.join(os.path.expanduser('~'), '/disk2/lyw/code/DANet-main')
OPTION.input_size = (241,425)
OPTION.test_size = (241,425)
OPTION.SNAPSHOT_DIR = os.path.join(OPTION.root_path, 'DANsnapshots')
