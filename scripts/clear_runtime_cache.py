"""Clear runtime Redis state after graph/state-shape changes.

This removes LangGraph checkpoints and short-lived retrieval caches only. It
does not remove schema metadata caches or business data.
"""

from __future__ import annotations

import argparse
import asyncio

import redis.asyncio as redis

from agents.config.settings import settings


def _redis_url() -> str:
    host, port = settings.redis.addr.rsplit(":", 1) if ":" in settings.redis.addr else (settings.redis.addr, "6379")
    if settings.redis.password:
        return f"redis://:{settings.redis.password}@{host}:{port}/{settings.redis.db}"
    return f"redis://{host}:{port}/{settings.redis.db}"


async def _delete_by_patterns(client: redis.Redis, patterns: list[str], dry_run: bool) -> dict[str, int]:
    deleted: dict[str, int] = {}
    for pattern in patterns:
        count = 0
        async for key in client.scan_iter(match=pattern, count=500):
            count += 1
            if not dry_run:
                await client.delete(key)
        deleted[pattern] = count
    return deleted


async def main() -> None:
    parser = argparse.ArgumentParser(description="Clear LangGraph checkpoints and retrieval caches from Redis")
    parser.add_argument("--dry-run", action="store_true", help="Only count matching keys")
    args = parser.parse_args()

    patterns = [
        "checkpoint:*",
        "checkpoint_write:*",
        "retrieval:*",
        "embedding:*",
    ]
    client = redis.from_url(_redis_url(), decode_responses=True)
    try:
        await client.ping()
        result = await _delete_by_patterns(client, patterns, dry_run=args.dry_run)
    finally:
        await client.aclose()

    action = "Matched" if args.dry_run else "Deleted"
    for pattern, count in result.items():
        print(f"{action} {count} keys for {pattern}")


if __name__ == "__main__":
    asyncio.run(main())
