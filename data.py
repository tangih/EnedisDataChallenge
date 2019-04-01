import time
import datetime

def timestamp2utc(t, min_timestamp=1381694400):
    timestamp = t + min_timestamp
    return datetime.datetime.fromtimestamp(timestamp).isoformat()
