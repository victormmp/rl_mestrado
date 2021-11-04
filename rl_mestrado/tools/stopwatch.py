"""Main module."""

import datetime


class StopWatch:
    start = None

    def __init__(self):
        self.reset()

    def reset(self):
        """
        Restart timer.

        Returns
        -------
            None.
        """
        self.start = datetime.datetime.now()

    def read(self, in_seconds=False):
        """
        Get total time since instantiation.

        Parameters
        ----------
        in_seconds: bool
            Flag to return data in seconds.

        Returns
        -------
        float
            Total time since object instantiation, in milliseconds if "in_seconds" is false. Else, return time in
            seconds.
        """
        delta_time = datetime.datetime.now() - self.start
        if in_seconds:
            return delta_time.total_seconds()
        return delta_time
