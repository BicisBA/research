import pathlib
import tempfile

import boto3

from models.config import cfg
import logging

logger = logging.getLogger(__name__)


def download_cert(
    endpoint_url: str, access_key: str, secret_key: str, bucket: str, prefix: str
) -> str:
    filename = prefix.split("/")[-1]
    cert_location = str(pathlib.Path().joinpath(tempfile.gettempdir(), filename))

    minio = boto3.resource(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        verify=False,
    )
    minio.Bucket(bucket).Object(prefix).download_file(cert_location)

    return str(cert_location)


class S3Client:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(S3Client, cls).__new__(cls)

            endpoint_url = cfg.s3.endpoint_url(default=None)
            if endpoint_url is not None:
                logger.info("S3 endpoint is %s", endpoint_url)

            access_key = cfg.s3.access_key()
            secret_key = cfg.s3.secret_key()

            bucket = cfg.s3.bucket()
            cls._instance.bucket = bucket

            s3_cert = cfg.s3.cert(default=None)

            if s3_cert is not None and s3_cert != "":
                logger.info(
                    "Looking for certificate in bucket %s at prefix %s", bucket, s3_cert
                )

                cert_location = download_cert(
                    endpoint_url, access_key, secret_key, bucket, s3_cert
                )

                cls._boto_client = boto3.resource(
                    "s3",
                    endpoint_url=endpoint_url,
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                    verify=cert_location,
                )
            else:
                cls._boto_client = boto3.resource(
                    "s3",
                    endpoint_url=endpoint_url,
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                    verify=False,
                )

        return cls._instance

    @property
    def client(self):
        return self._boto_client
