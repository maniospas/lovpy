"""ERROR EXAMPLE FROM: http://effbot.org/zone/thread-synchronization.htm"""
import threading


lock = threading.Lock()
obj = "ab"


def get_first_part():
    lock.acquire()
    try:
        data = obj[0]
    finally:
        lock.release()
    return data


def get_second_part():
    lock.acquire()
    try:
        data = obj[1]
    finally:
        lock.release()
    return data


def get_both_parts():
    # This will hang!
    lock.acquire()
    lock.release()
    lock.acquire()
    try:
        first = get_first_part()
        second = get_second_part()
    finally:
        pass
        # lock.release()
    return first, second


# threading.Thread(target=get_first_part).start()
# threading.Thread(target=get_second_part).start()
get_both_parts()
