import threading


class TimeSource:
    def __init__(self):
        self._current_time = 0
        self._lock = threading.Lock()

    def stamp_and_increment(self):
        self._lock.acquire()
        stamp = self._current_time
        self._current_time += 1
        self._lock.release()
        return stamp

    def get_current_time(self):
        return self._current_time

    # def pause(self):
    #     self._lock.acquire()
    #
    # def resume(self):
    #     self._lock.release()


global_time_source = TimeSource()


def get_global_time_source():
    return global_time_source


def global_stamp_and_increment():
    return global_time_source.stamp_and_increment()