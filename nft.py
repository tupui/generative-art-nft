import pathlib
from typing import List, Tuple

from PIL import Image
import pandas as pd
from progressbar import progressbar
import numpy as np
from scipy.stats import qmc

from config import CONFIG

# Initialize random number generator
rng = np.random.default_rng()

# Path constants
ASSETS_PATH = pathlib.Path("assets")
OUTPUT_PATH = pathlib.Path("output")


def parse_config() -> None:
    """Parse the configuration file and validate layer setup."""
    for layer in CONFIG:
        layer_path = ASSETS_PATH / layer["directory"]
        if not layer_path.exists():
            raise FileNotFoundError(f"Layer directory not found: {layer_path}")

        # Get trait files sorted by name
        traits = sorted([trait.name for trait in layer_path.glob("*.png")])

        # Add None option for optional layers
        if not layer["required"]:
            traits.insert(0, None)

        # Process rarity weights
        rarities = _process_rarity_weights(layer["rarity_weights"], traits)
        rarities = np.array(rarities) / sum(rarities)

        # Update layer config
        layer.update(
            {
                "rarity_weights": rarities,
                "cum_rarity_weights": np.cumsum(rarities),
                "traits": traits,
            }
        )


def _process_rarity_weights(rarity_config, traits: List) -> List[float]:
    """Process different rarity weight configurations."""
    if rarity_config is None:
        return [1.0] * len(traits)
    elif isinstance(rarity_config, list):
        if len(rarity_config) != len(traits):
            raise ValueError(
                f"Rarity weights length ({len(rarity_config)}) "
                f"doesn't match traits length ({len(traits)})"
            )
        return rarity_config
    else:
        raise ValueError(f"Invalid rarity weights: {rarity_config}")


def get_trait_indices(
    random_matrix: np.ndarray, cum_rarities_list: List[np.ndarray]
) -> np.ndarray:
    """Convert Halton random matrix to trait indices using inverse transform sampling.

    Uses cumulative rarity weights as CDFs for each layer dimension.
    np.searchsorted with side='right' performs inverse CDF sampling.

    Args:
        random_matrix: Halton random values (num_samples, num_layers) in [0,1)
        cum_rarities_list: Cumulative rarity weights for each layer

    Returns:
        Matrix of trait indices (num_samples, num_layers)
    """
    num_samples, num_layers = random_matrix.shape
    indices = np.zeros((num_samples, num_layers), dtype=int)

    for layer_idx in range(num_layers):
        rand_vals = random_matrix[:, layer_idx]
        cum_rarities = cum_rarities_list[layer_idx]
        indices[:, layer_idx] = np.searchsorted(cum_rarities, rand_vals, side="right")

    return indices


def generate_single_image(filepaths, output_filename=None):
    """Generate a single image given an array of filepaths representing layers."""
    # Treat the first layer as the background
    bg = Image.open(filepaths[0])

    # Loop through layers 1 to n and stack them on top of another
    for filepath in filepaths[1:]:
        img = Image.open(filepath)
        bg.paste(img, (0, 0), img)

    bg.save(output_filename)


def get_total_combinations():
    """Get total number of distinct possible combinations."""
    total = 1
    for layer in CONFIG:
        total = total * len(layer["traits"])
    return total


def generate_all_trait_sets(num_samples: int) -> List[Tuple[List, List[pathlib.Path]]]:
    """Generate unique trait combinations using Halton sequences with oversampling.

    Args:
        num_samples: Number of NFTs to generate

    Returns:
        List of (trait_names, trait_paths) tuples for each sample
    """
    num_layers = len(CONFIG)

    # Oversample to account for duplicates, cap at total possible combinations
    n_candidates = min(num_samples * 2, get_total_combinations())

    # Generate Halton quasi-random matrix
    sampler = qmc.Halton(d=num_layers, seed=rng)
    random_matrix = sampler.random(n=n_candidates)

    # Get cumulative distributions for inverse transform sampling
    cum_rarities_list = [layer["cum_rarity_weights"] for layer in CONFIG]

    # Convert uniform [0,1) values to trait indices
    indices_matrix = get_trait_indices(random_matrix, cum_rarities_list)

    # Build all trait combinations
    all_combinations = []
    for sample_idx in range(n_candidates):
        trait_names = []
        trait_paths = []

        for layer_idx in range(num_layers):
            trait_idx = indices_matrix[sample_idx, layer_idx]
            trait_name = CONFIG[layer_idx]["traits"][trait_idx]
            trait_names.append(trait_name)

            if trait_name is not None:
                trait_path = ASSETS_PATH / CONFIG[layer_idx]["directory"] / trait_name
                trait_paths.append(trait_path)

        all_combinations.append((trait_names, trait_paths))

    # Remove duplicates, keep first occurrence (preserves quasi-random order)
    df = pd.DataFrame(
        {"combo": [tuple(x[0]) for x in all_combinations], "data": all_combinations}
    )
    unique_combinations = df.drop_duplicates(subset="combo")["data"].tolist()
    result = unique_combinations[:num_samples]

    return result


def generate_images(edition: str, count: int) -> pd.DataFrame:
    """Generate NFT images and return metadata DataFrame.

    Args:
        edition: Edition name for output directory
        count: Number of images to generate

    Returns:
        DataFrame with trait metadata for each generated NFT
    """
    # Initialize rarity tracking
    rarity_data = {layer["name"]: [] for layer in CONFIG}

    # Create output directory
    images_dir = OUTPUT_PATH / f"edition_{edition}" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Calculate zero-padding width
    zfill_width = len(str(count - 1))

    # Generate all trait combinations upfront
    all_trait_sets = generate_all_trait_sets(count)

    # Generate images
    for idx in progressbar(range(count)):
        image_name = f"{idx:0{zfill_width}d}.png"
        trait_names, trait_paths = all_trait_sets[idx]

        # Create composite image
        generate_single_image(trait_paths, images_dir / image_name)

        # Record traits for metadata (remove .png extension)
        for layer_idx, trait_name in enumerate(trait_names):
            layer_name = CONFIG[layer_idx]["name"]
            clean_trait = trait_name.replace(".png", "") if trait_name else "none"
            rarity_data[layer_name].append(clean_trait)

    metadata_df = pd.DataFrame(rarity_data)
    print(f"Generated {count} images")

    return metadata_df


def generate_rarity_stats(metadata_df: pd.DataFrame) -> None:
    """Generate and display rarity statistics comparing actual vs target distributions."""
    for layer in CONFIG:
        layer_name = layer["name"]
        if layer_name not in metadata_df.columns:
            continue

        print(f"\n{layer_name.upper()}:")

        # Get target distribution (already computed during config parsing)
        target_dist = dict(
            zip(
                [t.replace(".png", "") if t else "none" for t in layer["traits"]],
                [float(w) for w in layer["rarity_weights"]],
            )
        )

        # Get actual distribution
        actual_dist = _get_actual_distribution(
            metadata_df[layer_name], target_dist.keys()
        )

        # Display results
        print(f"  Target: {target_dist}")
        print(f"  Actual: {actual_dist}")

        # Show per-trait differences
        max_diff = _print_trait_differences(target_dist, actual_dist)
        print(f"  Max difference: {max_diff:.4f}")


def _get_actual_distribution(series: pd.Series, expected_traits: set) -> dict:
    """Calculate actual trait distribution from metadata."""
    total_samples = len(series)
    actual_dist = {}

    for trait in expected_traits:
        count = (series == trait).sum()
        actual_dist[trait] = count / total_samples

    return actual_dist


def _print_trait_differences(target_dist: dict, actual_dist: dict) -> float:
    """Print per-trait differences and return maximum difference."""
    max_diff = 0.0

    for trait in target_dist.keys():
        target_prob = target_dist[trait]
        actual_prob = actual_dist[trait]
        diff = abs(actual_prob - target_prob)
        max_diff = max(max_diff, diff)
        print(
            f"    {trait}: {actual_prob:.4f} (target: {target_prob:.4f}, diff: {diff:.4f})"
        )

    return max_diff


def main() -> None:
    """Main NFT generation workflow."""
    print("Checking assets...")
    parse_config()
    print("✅ Assets validated successfully!\n")

    total_combinations = get_total_combinations()
    print(f"You can create up to {total_combinations} distinct avatars\n")

    # Get user input
    num_avatars = int(input("How many avatars would you like to create? "))
    edition_name = input("What would you like to call this edition?: ").strip()

    print("Starting generation...")
    metadata_df = generate_images(edition_name, num_avatars)

    print("Saving metadata...")
    metadata_path = OUTPUT_PATH / f"edition_{edition_name}" / "metadata.csv"
    metadata_df.to_csv(metadata_path)

    # Generate rarity statistics
    print("\n=== Rarity Statistics ===")
    generate_rarity_stats(metadata_df)

    print("✅ Task complete!")
    print(f"\n📝 Next step: Run 'python metadata.py' to generate JSON metadata files")


# Run the main function
if __name__ == "__main__":
    main()
