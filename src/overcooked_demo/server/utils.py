import os
from threading import Lock

# this is the mounted volume
DOCKER_VOLUME = "/app/data"


class ThreadSafeSet(set):
    def __init__(self, *args, **kwargs):
        super(ThreadSafeSet, self).__init__(*args, **kwargs)
        self.lock = Lock()

    def add(self, *args):
        with self.lock:
            retval = super(ThreadSafeSet, self).add(*args)
        return retval

    def clear(self, *args):
        with self.lock:
            retval = super(ThreadSafeSet, self).clear(*args)
        return retval

    def pop(self, *args):
        with self.lock:
            if len(self):
                retval = super(ThreadSafeSet, self).pop(*args)
            else:
                retval = None
        return retval

    def remove(self, item):
        with self.lock:
            if item in self:
                retval = super(ThreadSafeSet, self).remove(item)
            else:
                retval = None
        return retval


class ThreadSafeDict(dict):
    def __init__(self, *args, **kwargs):
        super(ThreadSafeDict, self).__init__(*args, **kwargs)
        self.lock = Lock()

    def clear(self, *args, **kwargs):
        with self.lock:
            retval = super(ThreadSafeDict, self).clear(*args, **kwargs)
        return retval

    def pop(self, *args, **kwargs):
        with self.lock:
            retval = super(ThreadSafeDict, self).pop(*args, **kwargs)
        return retval

    def __setitem__(self, *args, **kwargs):
        with self.lock:
            retval = super(ThreadSafeDict, self).__setitem__(*args, **kwargs)
        return retval

    def __delitem__(self, item):
        with self.lock:
            if item in self:
                retval = super(ThreadSafeDict, self).__delitem__(item)
            else:
                retval = None
        return retval


def create_dirs(config: dict, cur_layout: str):
    """
    config has 3 keys:
     {"time": datetime.today().strftime("%Y-%m-%d_%H-%M-%S"),
      "type": gameType/a str of either "HH","HA","AH","AA",
      "layout": a layout string}
    We group the data by layout/type/time
    """
    path = os.path.join(
        DOCKER_VOLUME,
        cur_layout,
        config["old_dynamics"],
        config["type"],
        config["time"],
    )
    if not os.path.exists(path):
        os.makedirs(path)
    return path
