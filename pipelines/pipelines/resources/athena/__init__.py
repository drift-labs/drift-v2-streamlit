from dagster import ConfigurableResource
from dagster_aws.athena import AthenaClient, AthenaClientResource

import pandas as pd
import boto3
import csv
from urllib.parse import urlparse


class AthenaConfig(ConfigurableResource):
    database_name: str
    trades_table_name: str = "eventtype_traderecord"
    orders_table_name: str = "eventtype_orderrecord"
    actions_table_name: str = "eventtype_orderactionrecord"
    dlob_snapshot_table_name: str = "eventtype_dlobsnapshot"


class WrappedAthenaClient(AthenaClient):
    def _results(self, execution_id):
        execution = self.client.get_query_execution(QueryExecutionId=execution_id)[
            "QueryExecution"
        ]
        s3 = boto3.resource("s3")
        output_location = execution["ResultConfiguration"]["OutputLocation"]
        bucket = urlparse(output_location).netloc
        prefix = urlparse(output_location).path.lstrip("/")

        rows = (
            s3.Bucket(bucket)
            .Object(prefix)
            .get()["Body"]
            .read()
            .decode("utf-8")
            .splitlines()
        )
        reader = csv.reader(rows)
        rows = list(reader)

        if not rows:
            return pd.DataFrame()

        header = rows[0]
        data_rows = rows[1:]

        return pd.DataFrame(data_rows, columns=header)


class WrappedAthenaClientResource(AthenaClientResource):
    def get_client(self) -> WrappedAthenaClient:
        client = boto3.client(
            "athena",
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
        )
        return WrappedAthenaClient(
            client=client,
            workgroup=self.workgroup,
            polling_interval=self.polling_interval,
            max_polls=self.max_polls,
        )
