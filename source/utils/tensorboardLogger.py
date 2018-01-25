
from tensorboardX import SummaryWriter


class TensorboardLogger:

    def __init__(self, log_dir: str):
        self._writer = SummaryWriter(log_dir=log_dir)

    def log_training_episode_data(self, episode: int, loss: float, epsilon: float):

        self._log_scalar(tag="loss_vs_episode", value=loss, step=episode)
        self._log_scalar(tag="epsilon_vs_episode", value=epsilon, step=episode)

    def log_transition_data(self, transition: int, training_episode: int, reward: float):
        self._log_scalar(tag="reward_vs_transition", value=reward, step=transition)
        self._log_scalar(tag="transition_vs_episode", value=transition, step=training_episode)

    def log_epsilon_0_game_summary(self, training_episode: int, final_score: int):
        self._log_scalar(tag="epsilon_0_game_score_vs_episode", value=final_score, step=training_episode)

    def _log_scalar(self, tag: str, value, step: int):
        """Log a scalar variable.
        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value
        step : int
            training iteration
        """
        self._writer.add_scalar(tag="training/"+tag, scalar_value=value, global_step=step)
