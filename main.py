
import argparse
import config
from src.evaluation.runner import run_evaluation
from src.reporting import generate_leaderboard

def main():
    # Load env vars
    from dotenv import load_dotenv
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Korean LLM Benchmark Evaluation")
    parser.add_argument("--dry-run", action="store_true", help="Run a test evaluation with minimal samples")
    parser.add_argument("--test-connection", action="store_true", help="Test API connections")
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("Dry run mode enabled. Setting sample size to 1.")
        config.SAMPLE_SIZE = 1
    
    if args.test_connection:
        print("Testing API connections...")
        # Simple test logic or calling a test function
        # For now, just print config
        print(f"Enabled Models: {config.ENABLED_MODELS}")
        return

    run_evaluation()
    generate_leaderboard()

if __name__ == "__main__":
    main()
