# Hazard Image Generation — Instructions

## What This Does
Generates hazard-injected warehouse images using Gemini's image editing via Playwright browser automation. Each run takes a clean warehouse render, sends it to Gemini with a hazard injection prompt, and downloads the modified image.

## One-Time Setup (~5 min)

### 1. Clone the repo
```bash
git clone <repo-url>
cd safety-hazard-detection
```

### 2. Install dependencies
```bash
pip3 install playwright
playwright install chromium
```

### 3. Launch Chrome with remote debugging
```bash
# macOS
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222

# Windows
"C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222

# Linux
google-chrome --remote-debugging-port=9222
```

### 4. Sign into Gemini
In the Chrome window that opened, go to https://gemini.google.com and sign in with your Google account.

---

## Generation Commands

Run these one at a time. Each command will print progress and auto-retry on failures. You can stop anytime with Ctrl+C — it saves progress and skips already-generated images on restart.

### Forklift Violations (~57 images, ~90 min)
```bash
python3 scripts/gemini_hazard_inject.py --category forklift_violation --start 0 --end 170 --step 3 --overnight
```

### Obstacles (~57 images, ~90 min)
```bash
python3 scripts/gemini_hazard_inject.py --category obstacle --start 0 --end 170 --step 3 --overnight
```

### More Spills (~57 images, ~90 min)
Uses frames not already used (offset by 1):
```bash
python3 scripts/gemini_hazard_inject.py --category spill --start 1 --end 170 --step 3 --overnight
```

### More Improper Stacking (~57 images, ~90 min)
Uses frames not already used (offset by 2):
```bash
python3 scripts/gemini_hazard_inject.py --category improper_stacking --start 2 --end 170 --step 3 --overnight
```

---

## Notes
- **Rate limits**: If Gemini rate-limits you (3 failures in a row), the script auto-pauses for 10 minutes then resumes.
- **Resume safe**: The script checks which images already exist and skips them. You can stop and restart anytime.
- **Output**: Images saved to `outputs/datasets/images/<category>/`
- **Labels**: Metadata saved to `outputs/datasets/labels.json`
- **Logs**: Check `hazard_injection.log` for detailed progress

## What We Need Total

| Category | Have | Need | Command |
|----------|------|------|---------|
| spill | ~70 | ~57 more | spill command above |
| improper_stacking | ~113 | ~57 more | stacking command above |
| forklift_violation | 0 | ~57 | forklift command above |
| obstacle | 0 | ~57 | obstacle command above |
| safe | 170 | 0 | Already have base renders |

Priority order: **forklift_violation → obstacle → spill → improper_stacking**
