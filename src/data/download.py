"""Download the IBM Telco Customer Churn dataset into data/raw/."""

from __future__ import annotations

import ssl
import urllib.error
import urllib.request
from pathlib import Path

from .load import load_config, project_root

# Public mirror of the IBM sample Telco Customer Churn dataset
TELCO_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)


def download_telco(destination: Path | None = None, url: str = TELCO_URL) -> Path:
    """Download the Telco CSV if it is not already present."""
    root = project_root()
    if destination is None:
        cfg = load_config()
        destination = root / cfg["paths"]["raw_data"]
    else:
        destination = Path(destination)
        if not destination.is_absolute():
            destination = root / destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Dataset already present at {destination}")
        return destination

    print(f"Downloading Telco churn data from {url}")
    # Some local Python installs lack system CA certs; fall back gracefully.
    try:
        urllib.request.urlretrieve(url, destination)
    except urllib.error.URLError:
        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(url, context=ctx) as resp, open(destination, "wb") as out:
            out.write(resp.read())
    print(f"Saved to {destination}")
    return destination


def main() -> None:
    download_telco()


if __name__ == "__main__":
    main()
