"""Quick E2E diagnostic: capture screenshot, send to Gemini, print raw response"""
import sys, os, logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

import config
from vision.screen_capture import ScreenCapture
from vision.gemini_analyzer import GeminiAnalyzer

# 1. Capture
sc = ScreenCapture(config.WINDOW_TITLE_PATTERN)
if not sc.find_window():
    print("ERROR: MuMuPlayer window not found")
    sys.exit(1)

dims = sc.get_game_dimensions()
print(f"Window dimensions: {dims}")
print(f"Game area: {sc.game_area}")

# Save screenshot for inspection
pil_img = sc.capture_pil()
if pil_img is None:
    print("ERROR: capture failed")
    sys.exit(1)

pil_img.save("_debug_screenshot.png")
print(f"Screenshot saved: _debug_screenshot.png ({pil_img.size})")

# 2. Send to Gemini and print raw response
from google import genai
client = genai.Client(api_key=config.GEMINI_API_KEY)

prompt = GeminiAnalyzer.SYSTEM_PROMPT + "\n\n이미지를 분석하고 JSON 형식으로 결과를 반환하라."
r = client.models.generate_content(model=config.GEMINI_MODEL, contents=[prompt, pil_img])
print(f"\n=== RAW GEMINI RESPONSE ===\n{r.text}\n=== END ===")

# 3. Parse
analyzer = GeminiAnalyzer(api_key=config.GEMINI_API_KEY, model_name=config.GEMINI_MODEL)
coins, next_coin, game_area = analyzer.analyze_and_extract(pil_img)
print(f"\nParsed: {len(coins)} coins, next={next_coin}, area={game_area}")
for c in coins:
    print(f"  {c.coin_type.display_name} at ({c.x:.0f}, {c.y:.0f})")
