from trainer import Trainer


class TrainerDelegate:

    def __init__(self):
        pass

    def trainer_did_finish_training(self, trainer: Trainer):
        raise NotImplementedError

    def trainer_did_finish_training_episode(self, trainer: Trainer, episode: int):
        pass