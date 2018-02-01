import importlib
import ntpath
import os

import keras
import sys

from source.utils.previewHelper import PreviewHelper
from source.utils.utils import get_options

if __name__ == '__main__':
    args = get_options(sys.argv)

    model_file = args["-model"]
    env_file = args["-env-file"]
    env_class_name = args["-env-class"]
    num_episodes = int(args["-n"])

    env_module_name = os.path.splitext(ntpath.basename(env_file))[0]
    loader = importlib.machinery.SourceFileLoader(env_module_name, env_file)
    env_module = loader.load_module(env_module_name)

    EnvClass = getattr(env_module, env_class_name)
    preview_helper = PreviewHelper(game=EnvClass(), model=keras.models.load_model(filepath=model_file))
    preview_helper.play(episodes=num_episodes, display=True)