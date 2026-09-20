"""Fetch the historical SSE daily chart; do not replace an existing cache silently."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from benchmark_support import DATA_PATH, data_fingerprint, parse_prices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Explicitly replace the cached response")
    parser.add_argument("--from-file", type=Path, help="Validate and import your existing Yahoo chart JSON")
    args = parser.parse_args()
    target = DATA_PATH
    if target.exists() and not args.force:
        print(f"Existing cache retained. Canonical prices SHA256: {data_fingerprint()}")
        return
    start = int(datetime(2023, 3, 1, tzinfo=timezone.utc).timestamp())
    end = int(datetime(2026, 9, 12, tzinfo=timezone.utc).timestamp())
    url = ("https://query1.finance.yahoo.com/v8/finance/chart/000001.SS"
           f"?period1={start}&period2={end}&interval=1d")
    if args.from_file:
        content = args.from_file.read_bytes()
        source = "user-supplied saved Yahoo chart response"
    else:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=45) as response:
                    content = response.read()
                break
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt == 2:
                    raise SystemExit(
                        f"Data download failed: {exc}. Retry later or use --from-file."
                    ) from exc
                time.sleep(2 * (attempt + 1))
        source = url
    payload = json.loads(content.decode("utf-8"))
    prices = parse_prices(payload)
    if prices.index.min() > datetime(2023, 3, 14) or prices.index.max().strftime("%Y-%m-%d") != "2026-09-11":
        raise SystemExit("Data does not cover the full published experiment dates.")
    if len(prices.loc["2026-06-15":"2026-09-11"]) != 64:
        raise SystemExit("Expected 64 observed review dates; check the provider's response.")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".json.part")
    temp.write_bytes(content)
    temp.replace(target)
    fingerprint = data_fingerprint()
    metadata = {"source": source, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "symbol": "000001.SS", "observations": len(prices),
                "first": str(prices.index.min().date()), "last": str(prices.index.max().date()),
                "response_sha256": hashlib.sha256(content).hexdigest(),
                "canonical_prices_sha256": fingerprint,
                "note": "Provider revisions can change a rerun; archived reference results remain separate."}
    target.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {len(prices)} observations. Canonical prices SHA256: {fingerprint}")


if __name__ == "__main__":
    main()
