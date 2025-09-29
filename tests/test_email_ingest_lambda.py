import io
import json
from email.message import EmailMessage

import boto3
import pytest
from botocore.stub import Stubber

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from services import email_ingest_lambda as ingest


def _build_email(subject="RFQ-20240101-abc12345", body="Quote 1200", **headers):
    message = EmailMessage()
    message["Subject"] = subject
    for key, value in headers.items():
        message[key] = value
    message.set_content(body)
    return message.as_bytes()


_ACTIVE_STUBBERS = []


def _prepare_s3_stub(email_bytes, bucket, key, *, rfq="RFQ-20240101-ABC12345", unmatched=False):
    client = boto3.client("s3", region_name="eu-west-1")
    stub = Stubber(client)

    stub.add_response(
        "get_object",
        {"Body": io.BytesIO(email_bytes)},
        {"Bucket": bucket, "Key": key},
    )

    if unmatched:
        dest_key = f"emails/_unmatched/{key.split('/')[-1]}"
        stub.add_response(
            "copy_object",
            {},
            {
                "Bucket": bucket,
                "CopySource": {"Bucket": bucket, "Key": key},
                "Key": dest_key,
                "TaggingDirective": "REPLACE",
                "Tagging": "needs-review=true",
            },
        )
    else:
        stub.add_response(
            "put_object_tagging",
            {},
            {
                "Bucket": bucket,
                "Key": key,
                "Tagging": {"TagSet": [{"Key": "rfq-id", "Value": rfq}]},
            },
        )
        dest_key = f"emails/{rfq}/ingest/{key.split('/')[-1]}"
        stub.add_response(
            "copy_object",
            {},
            {
                "Bucket": bucket,
                "CopySource": {"Bucket": bucket, "Key": key},
                "Key": dest_key,
                "TaggingDirective": "REPLACE",
                "Tagging": f"rfq-id={rfq}",
            },
        )

    stub.activate()
    _ACTIVE_STUBBERS.append(stub)
    ingest._S3_CLIENT = client
    return client, stub


class _FakeTable:
    def __init__(self, mapping):
        self.mapping = mapping

    def get_item(self, Key):
        message_id = Key.get("message_id")
        if message_id in self.mapping:
            return {"Item": {"rfq_id": self.mapping[message_id]}}
        return {}


def test_process_record_tags_and_copies_object(monkeypatch):
    bucket = "procwisemvp"
    key = "emails/random-object"
    email_bytes = _build_email(subject="Re: RFQ-20240101-abc12345", body="Price 1200")

    _prepare_s3_stub(email_bytes, bucket, key)
    ingest._DDB_TABLE = _FakeTable({})

    record = {"s3": {"bucket": {"name": bucket}, "object": {"key": key}}}
    result = ingest.process_record(record)

    assert result == {
        "rfq_id": "RFQ-20240101-ABC12345",
        "s3_key": "emails/RFQ-20240101-ABC12345/ingest/random-object",
        "status": "ok",
    }


def test_process_record_uses_thread_lookup(monkeypatch):
    bucket = "procwisemvp"
    key = "emails/random-object-2"
    email_bytes = _build_email(subject="Re: Quote", body="No RFQ", **{"In-Reply-To": "<thread-123>"})

    _prepare_s3_stub(email_bytes, bucket, key, rfq="RFQ-20240202-THREAD12")
    ingest._DDB_TABLE = _FakeTable({"<thread-123>": "RFQ-20240202-THREAD12"})

    record = {"s3": {"bucket": {"name": bucket}, "object": {"key": key}}}
    result = ingest.process_record(record)

    assert result["rfq_id"] == "RFQ-20240202-THREAD12"
    assert result["status"] == "ok"
    assert result["s3_key"].endswith("random-object-2")


def test_process_record_moves_to_unmatched_when_missing_rfq():
    bucket = "procwisemvp"
    key = "emails/random-object-3"
    email_bytes = _build_email(subject="Quote", body="No identifier present")

    _prepare_s3_stub(email_bytes, bucket, key, unmatched=True)
    ingest._DDB_TABLE = _FakeTable({})

    record = {"s3": {"bucket": {"name": bucket}, "object": {"key": key}}}
    result = ingest.process_record(record)

    assert result == {
        "rfq_id": None,
        "s3_key": "emails/_unmatched/random-object-3",
        "status": "needs_review",
    }


def test_lambda_handler_processes_sqs_envelope(monkeypatch):
    bucket = "procwisemvp"
    key = "emails/random-object-4"
    email_bytes = _build_email(subject="Re: RFQ-20240102-beefcafe", body="Confirming order")

    _prepare_s3_stub(email_bytes, bucket, key, rfq="RFQ-20240102-BEEFCAFE")
    ingest._DDB_TABLE = _FakeTable({})

    s3_record = {
        "eventSource": "aws:s3",
        "s3": {"bucket": {"name": bucket}, "object": {"key": key}},
    }
    sqs_event = {
        "Records": [
            {
                "body": json.dumps({"Records": [s3_record]}),
            }
        ]
    }

    response = ingest.lambda_handler(sqs_event, context=None)

    assert response["processed"][0]["rfq_id"] == "RFQ-20240102-BEEFCAFE"


@pytest.fixture(autouse=True)
def reset_clients():
    yield
    ingest._S3_CLIENT = None
    ingest._DDB_TABLE = None
    while _ACTIVE_STUBBERS:
        stub = _ACTIVE_STUBBERS.pop()
        stub.deactivate()

