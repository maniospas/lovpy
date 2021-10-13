import threading
import types


class TimeSource:
    def __init__(self):
        self._current_time = 0
        self._lock = threading.Lock()

    def stamp_and_increment(self):
        self._lock.acquire()
        self._current_time += 1
        stamp = self._current_time
        self._lock.release()
        return stamp

    def get_current_time(self):
        return self._current_time

    def __copy__(self):
        new_timesource = type(self)()
        new_timesource._current_time = self._current_time
        return new_timesource

    def __deepcopy__(self, memodict={}):
        new_timesource = type(self)()
        new_timesource._current_time = self._current_time
        return new_timesource

    # def pause(self):
    #     self._lock.acquire()
    #
    # def resume(self):
    #     self._lock.release()


global_time_source = TimeSource()
zero_locked_time_source = None


def get_global_time_source():
    return global_time_source


def global_stamp_and_increment():
    return global_time_source.stamp_and_increment()


def get_zero_locked_timesource():
    global zero_locked_time_source

    if not zero_locked_time_source:
        zero_locked_time_source = TimeSource()

        def get_zero(self):
            return 0

        zero_locked_time_source.get_current_time = types.MethodType(
            get_zero, zero_locked_time_source)

    return zero_locked_time_source
