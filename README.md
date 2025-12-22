# Generative NFT Art

Generates images and metadata for NFT collections from asset layers.

## Installation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

## Usage

### Configuration

Edit `config.py` to define layers and rarity weights and place layer images in `assets/` directory.

### Generate Images
```bash
uv run python nft.py
```
Creates composite images using quasi-Monte Carlo sampling for rarity distribution and generates a metadata CSV.

### Generate Metadata
```bash
uv run python metadata.py
```
Creates individual JSON metadata files following OpenSea standard.

## Output

Files saved to `output/edition_{name}/`:
- `images/` - PNG files
- `metadata/` - JSON metadata files
- `metadata.csv` - Trait data
