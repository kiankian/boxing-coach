import google.generativeai as genai
import os
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure API key from environment variable
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

genai.configure(api_key=api_key)

# Initialize the model
model = genai.GenerativeModel('gemini-2.0-flash')

# Upload the punching video to Gemini
myfile = genai.upload_file("punching.mp4")
print(f"Uploaded file: {myfile.name}")

# Wait for the file to be processed
print("Waiting for file to be processed...")
while myfile.state.name == "PROCESSING":
    print(".", end="", flush=True)
    time.sleep(2)
    myfile = genai.get_file(myfile.name)

if myfile.state.name == "FAILED":
    raise ValueError(f"File processing failed: {myfile.state}")

print(f"\nFile is ready: {myfile.state.name}")

# Analyze the boxing video with a comprehensive prompt
response = model.generate_content([myfile, """
This is me boxing with a heavy bag.
Analyze the video at 1 fps.
Create one entry per bag impact (or clear miss/glance).

Each entry should include:
- timestamp_of_outcome — (M:SS.s)
- result — "landed", "glancing", or "missed"
- punch_type — "jab", "cross", "lead hook", "rear hook", "lead uppercut", "rear uppercut", "overhand", "body jab", "body hook", etc.
- bag_zone — "high", "mid", or "low"
- feedback — form, power mechanics, range, wrist alignment, recovery. Only if needed, max one every 3 seconds - otherwise give an empty string

Track running totals (include after feedback):
- total_good_punches (clean, powerful, well-executed punches)
- total_bad_punches (sloppy, weak, or poorly executed punches)

Output ONLY the raw JSON object under the top-level key "punches". No code fences, no markdown, no extra commentary.
"""]
)

print(response.text)