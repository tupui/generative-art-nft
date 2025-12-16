import json
import pathlib
from typing import Dict, Any
import pandas as pd
from copy import deepcopy
from progressbar import progressbar

# Configuration - EDIT THESE VALUES BEFORE RUNNING
BASE_IMAGE_URL = "ipfs://<-- Your CID Code-->"
BASE_NAME = "Bobiboum"
DESCRIPTION = ""

# Constants
NONE_VALUE = "none"
OUTPUT_DIR = pathlib.Path("output")


def create_base_metadata() -> Dict[str, Any]:
    """Create base metadata template."""
    return {
        "name": BASE_NAME,
        "description": DESCRIPTION,
        "image": BASE_IMAGE_URL,
        "attributes": [],
    }


def clean_column_name(name: str) -> str:
    """Convert snake_case to Title Case."""
    return name.replace("_", " ").title()


def load_metadata(edition_path: pathlib.Path) -> pd.DataFrame:
    """Load and clean metadata DataFrame."""
    metadata_path = edition_path / "metadata.csv"
    df = pd.read_csv(metadata_path)

    # Remove pandas index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)

    # Clean column names
    df.columns = [clean_column_name(col) for col in df.columns]

    return df


def generate_json_metadata(
    edition_path: pathlib.Path, metadata_df: pd.DataFrame
) -> pathlib.Path:
    """Generate JSON metadata files for all NFTs."""
    metadata_dir = edition_path / "metadata"
    metadata_dir.mkdir(exist_ok=True)

    base_metadata = create_base_metadata()
    zfill_width = len(str(len(metadata_df) - 1))

    print(f"Generating JSON metadata for {len(metadata_df)} NFTs...")

    for idx, row in progressbar(metadata_df.iterrows()):
        # Create item metadata
        item_metadata = deepcopy(base_metadata)
        item_metadata["name"] = f"{BASE_NAME} #{idx}"
        item_metadata["image"] = f"{BASE_IMAGE_URL}/{idx:0{zfill_width}d}.png"

        # Add traits (skip None values)
        for trait_type, trait_value in row.items():
            if trait_value != NONE_VALUE:
                item_metadata["attributes"].append(
                    {"trait_type": trait_type, "value": trait_value}
                )

        json_file = metadata_dir / str(idx)
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(item_metadata, f, indent=2, ensure_ascii=False)

    return metadata_dir


def main() -> None:
    """Generate JSON metadata files for NFT collection."""
    edition_name = input("Enter edition name to generate metadata for: ").strip()
    edition_path = OUTPUT_DIR / f"edition_{edition_name}"

    metadata_df = load_metadata(edition_path)
    metadata_dir = generate_json_metadata(edition_path, metadata_df)

    print(f"✅ Metadata generated in {metadata_dir}")


if __name__ == "__main__":
    main()
