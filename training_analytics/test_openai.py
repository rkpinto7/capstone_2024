import os
import logging
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_openai_api():
    """Test the OpenAI API connection and functionality."""
    try:
        # Initialize the OpenAI client
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY")  # Ensure this environment variable is set
        )

        prompt = f"""
        As an expert working in the PA Department of Labor and Industry, analyze this data and provide key insights:

        Attendance:
        - Average attendance: 63.73 participants
        - Best performing day: Saturday (83.0 avg participants)
        """

        # Create a chat completion
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=50
        )
        print("API Response:", response.choices[0].message.content)

    except Exception as e:
        logger.error(f"API call failed: {e}")

if __name__ == "__main__":
    test_openai_api()
