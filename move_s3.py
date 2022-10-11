from datetime import datetime as dt
import logging
from logging.handlers import RotatingFileHandler
import multiprocessing as mp
import operator as ops
import re

import boto3

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


file_handler = RotatingFileHandler("move_s3.log", mode='a', maxBytes=50 * 1024, backupCount=2, encoding=None, delay=0)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def move_obj(key):
    try:
        session = boto3.session.Session()
        s3 = session.resource("s3")
        bucket = s3.Bucket("babikesdata")

        ts = re.match(r"data/status/babikes_data_(?P<ts>\d+).json", key).group("ts")
        archive_datetime = dt.fromtimestamp(int(ts))
        
        new_datetime_prefix = archive_datetime.strftime("%Y/%-m/%-d/%-H/%M")
        
        new_key = f"data/silver/status/{new_datetime_prefix}.json"
        
        s3.meta.client.copy({"Bucket": bucket.name, "Key": key}, bucket.name, new_key)
        
        logger.info("Moved from %s to %s", key, new_key)
    except:
        logger.error("Error moving", exc_info=True)


def main():
    s3 = boto3.resource("s3")
    bucket = s3.Bucket("babikesdata")

    with mp.pool.ThreadPool(32) as pool:
        archives = map(ops.attrgetter("key"), bucket.objects.filter(Prefix="data/status/"),)
        results_pool = pool.imap_unordered(move_obj, archives, 8)
        for _ in results_pool:
            pass


if __name__ == "__main__":
    main()
