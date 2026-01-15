import argparse
import config

def main():
    parser = argparse.ArgumentParser(description="Korean LLM Benchmark Evaluation")
    parser.add_argument("--dry-run", action="store_true", help="Run a test evaluation with minimal samples")
    parser.add_argument("--test-connection", action="store_true", help="Test API connections")
    
    args = parser.parse_args()
    
    if args.test_connection:
        print("Testing API connections...")
        # TODO: Implement connection test
        return

    print(f"Starting evaluation with {len(config.ENABLED_MODELS)} models on {len(config.ENABLED_BENCHMARKS)} benchmarks.")
    if args.dry_run:
        print("Dry run mode enabled.")

if __name__ == "__main__":
    main()
