import argparse

from src.feature_extractor import FeatureExtractor
from src.utils import get_file_paths, save_json, load_image

EXTENSIONS = (".jpg", ".png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="Path to input dataset.")
    ap.add_argument("-o", "--output", default="library.json", help="Output path.")
    args = ap.parse_args()

    extractor = FeatureExtractor()

    paths = get_file_paths(args.dataset, EXTENSIONS)
    library_features, valid_paths = [], []
    for i, p in enumerate(paths):
        print(f"{i + 1}/{len(paths)}")
        image = load_image(p)
        if image is not None:
            feature_vector = extractor.extract(image)
            library_features.append(feature_vector)
            valid_paths.append(p)

    library = {}
    library["features"] = library_features
    library["paths"] = valid_paths
    save_json(args.output, library)


if __name__ == "__main__":
    main()
