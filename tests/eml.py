import os, sys, re, email, argparse
import boto3
from botocore.config import Config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--prefix", default="emails/")
    ap.add_argument("--max", type=int, default=20, help="max objects to show")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--region", default=None)
    args = ap.parse_args()

    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    s3 = session.client("s3", config=Config(retries={"max_attempts": 5}))

    print(f"Listing s3://{args.bucket}/{args.prefix} ...")
    paginator = s3.get_paginator("list_objects_v2")
    seen = 0
    pages = paginator.paginate(Bucket=args.bucket, Prefix=args.prefix)
    latest = []
    for page in pages:
        for obj in page.get("Contents", []):
            latest.append(obj)
    latest.sort(key=lambda x: (x["LastModified"], x["Key"]), reverse=True)
    latest = latest[:args.max]

    if not latest:
        print("No objects found under prefix.")
        return 2

    for obj in latest:
        key = obj["Key"]
        lm = obj["LastModified"]
        size = obj["Size"]
        print(f"\n--- {key}  ({size} bytes)  LastModified={lm} ---")
        try:
            resp = s3.get_object(Bucket=args.bucket, Key=key, Range="bytes=0-4095")
            blob = resp["Body"].read()
            msg = email.message_from_bytes(blob)
            def hdr(name): 
                return (msg.get(name) or "").strip().replace("r"," ").replace("n"," ")
            print("Subject:     ", hdr("Subject"))
            print("To:          ", hdr("To"))
            print("From:        ", hdr("From"))
            print("Message-ID:  ", hdr("Message-ID") or hdr("Message-Id"))
            print("In-Reply-To: ", hdr("In-Reply-To"))
            print("References:  ", hdr("References"))
            print("Content-Type:", hdr("Content-Type"))
        except Exception as e:
            print("Header sniff failed:", e)

    return 0

if __name__ == "__main__":
    sys.exit(main())