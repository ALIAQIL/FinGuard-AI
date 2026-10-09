"""Replay the real ULB CSV into Kafka in bounded batches."""

from __future__ import annotations

import argparse
import csv
import time

from finguard.streaming.events import event_from_row


def replay(input_path: str, topic: str, bootstrap_servers: str, rate: float = 10.0, limit: int | None = None) -> int:
    try:
        from kafka import KafkaProducer
    except ImportError as exc:  # pragma: no cover - optional runtime
        raise RuntimeError("Install kafka-python to run the replay producer") from exc
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        key_serializer=lambda value: value.encode(),
        value_serializer=lambda value: value.encode(),
        acks="all",
        retries=3,
        enable_idempotence=True,
    )
    sent = 0
    delay = 1.0 / rate if rate > 0 else 0.0
    with open(input_path, newline="", encoding="utf-8") as handle:
        for source_row_number, row in enumerate(csv.DictReader(handle), start=1):
            event = event_from_row(row, source_row_number)
            payload = event.to_json()
            producer.send(topic, key=event.event_id, value=payload)
            sent += 1
            if delay:
                time.sleep(delay)
            if limit and sent >= limit:
                break
    producer.flush()
    producer.close()
    return sent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw/creditcard.csv")
    parser.add_argument("--topic", default="transactions")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--rate", type=float, default=10.0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(f"sent={replay(args.input, args.topic, args.bootstrap_servers, args.rate, args.limit)}")


if __name__ == "__main__":
    main()

