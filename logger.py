from macnetwork.config import config
import tensorflow as tf

class Logger():
    """
    Define a logging facility for
    """
    def __init__(self,split,subsplit,current_time):
        """
        name: of log data
        """
        # current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = 'logs/{}/'.format(config.expName) +\
            current_time + '/{}'.format(split) +\
            '/{}'.format(subsplit)
        self.swriter = tf.summary.FileWriter(log_dir)

    def put(self,summary_info, iter_value=0):
        """
        put information into writing folder
        """
        step_idx = iter_value
        for summary in summary_info:
            self.swriter.add_summary(
                summary,
                global_step=step_idx)
