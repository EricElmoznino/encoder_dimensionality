from time import time
import datetime


def properties_to_id(architecture, task, kind, source):
    identifier = f'architecture:{architecture}|task:{task}|kind:{kind}|source:{source}'
    return identifier


def id_to_properties(identifier):
    identifier = identifier.split(',')[0]
    properties = identifier.split('|')[:4]
    properties = {p.split(':')[0]: p.split(':')[1] for p in properties}
    return properties


def timed(func):
    """Decorator that reports the execution time."""
    def wrap(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        elapsed = str(datetime.timedelta(seconds=(end - start)))
        print(f'{func.__name__} total runtime: {elapsed}')
        return result
    return wrap
