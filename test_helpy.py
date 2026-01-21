import os
from dotenv import load_dotenv
from src.models import HelpyEduModel

# Load environment variables
load_dotenv()

def test_helpy_edu():
    print('=== Testing Helpy Edu DragonFruit ===')
    try:
        # Check API key first
        api_key = os.getenv("ELICE_API_KEY")
        if not api_key:
            print("WARNING: ELICE_API_KEY not found in env")
            
        model = HelpyEduModel('helpy-edu')
        print("Sending request: 'Hi'...")
        response = model.generate('Hi', max_tokens=100)
        print(f"\nResponse:\n{response}")
        print("\n=== Test Complete ===")
    except Exception as e:
        print(f"\nError occurred: {e}")

if __name__ == "__main__":
    test_helpy_edu()
