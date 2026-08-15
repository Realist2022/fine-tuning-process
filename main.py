import argparse
import importlib


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CV fine-tuning pipeline.")
    parser.add_argument(
        "stage",
        choices=(
            "extract",
            "synthesize",
            "validate",
            "profile",
            "split",
            "train",
            "evaluate",
        ),
        help="Pipeline stage to run",
    )
    args = parser.parse_args()

    module_name, function_name = {
        "extract": ("01_extract", "extract_netsol_data"),
        "synthesize": ("02_synthesize", "generate_synthetic_data"),
        "validate": ("03_validate", "validate_and_format"),
        "profile": ("04_profile_tokens", "profile_dataset_tokens"),
        "split": ("05_split", "split_dataset"),
        "train": ("06_train", "run_training"),
        "evaluate": ("07_evaluate", "evaluate_model"),
    }[args.stage]
    getattr(importlib.import_module(module_name), function_name)()


if __name__ == "__main__":
    main()
