from logipy.wrappers import LogipyPrimitive, logipy_call
import threading

# ERROR EXAMPLE FROM: http://effbot.org/zone/thread-synchronization.htm

lock = logipy_call(threading.Lock,)

obj = "ab"

def get_first_part():
    logipy_call(lock.acquire,)
    try:
        data = obj[0]
    finally:
        logipy_call(lock.release,)
    return data

def get_second_part():
    logipy_call(lock.acquire,)
    try:
        data = obj[1]
    finally:
        logipy_call(lock.release,)
    return data

def get_both_parts():
    # THIS WILL HANG
    logipy_call(lock.acquire,)
    try:
        logipy_call(lock.release,)
        first = logipy_call(get_first_part,)
        second = logipy_call(get_second_part,)
    finally:
        logipy_call(lock.release,)
    return first, second

#logipy_call(threading.Thread,target=get_first_part).start()
#logipy_call(threading.Thread,target=get_second_part).start()
logipy_call(get_both_parts,)