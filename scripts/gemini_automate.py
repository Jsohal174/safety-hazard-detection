"""
Automate Gemini web app for photorealistic image conversion.
Connects to your real Chrome browser (with your Google login).

SETUP (one-time):
  1. Quit Chrome completely (Cmd+Q)
  2. Run this in terminal first:
     /Applications/Google Chrome.app/Contents/MacOS/Google Chrome --remote-debugging-port=9222
  3. Chrome opens with your profile. Log into Gemini if needed.
  4. Then run this script in another terminal tab.

Usage:
    python scripts/gemini_automate.py --start 73 --end 77
    python scripts/gemini_automate.py --start 73 --end 195 --overnight
"""

import argparse
import glob
import os
import time
import datetime
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE_DIR = "/Users/jaskiratsinghsohal/Desktop/safety-hazard-detection/outputs/renders/base"
OUTPUT_DIR = "/Users/jaskiratsinghsohal/Desktop/safety-hazard-detection/realistic"
LOG_FILE = "/Users/jaskiratsinghsohal/Desktop/safety-hazard-detection/gemini_automation.log"

FORKLIFT_PROMPT = """CRITICAL: This is a layout-preservation task. The warehouse structure, shelf positions, aisle width, camera angle, and every object's exact placement must remain IDENTICAL to the input image. Do NOT redesign, rearrange, or reinterpret the layout. Only change textures, materials, and lighting to make it photorealistic.

Transform this 3D warehouse render into a photorealistic photograph. The geometry and composition must be a 1:1 match to the input — same number of shelves, same aisle width, same camera height and angle, same box positions.

Make ONLY these material/texture changes:

- Tall pallet racking: orange painted uprights and blue/grey crossbeams with scratches and industrial wear.
- Fill ALL pallet rack frames on BOTH sides with loaded skids — heavy pallets stacked with shrink-wrapped boxes, industrial goods, and crates on every level of every rack.
- Concrete floor with realistic tire marks, scuff marks, and slight oil stains.
- Overhead fluorescent warehouse lighting casting harsh downward shadows.
- The forklift should look like a real used warehouse forklift with yellow paint, wear marks, and rubber tires.
- Cardboard boxes on pallets should have shipping labels, barcodes, and packing tape.
- Some pallets wrapped in clear/white stretch wrap with realistic reflections.
- Slight atmospheric dust haze in the air typical of large warehouses.

Style: photorealistic industrial photography, overhead drone security camera, large distribution warehouse. Output must be a texture upgrade of the input, NOT a new image."""

SHELVING_PROMPT = """CRITICAL: This is a layout-preservation task. The warehouse structure, shelf positions, aisle width, camera angle, and every object's exact placement must remain IDENTICAL to the input image. Do NOT redesign, rearrange, or reinterpret the layout. Only change textures, materials, and lighting to make it photorealistic.

Transform this 3D warehouse render into a photorealistic photograph. The geometry and composition must be a 1:1 match to the input — same number of shelves, same aisle width, same camera height and angle, same box positions.

Make ONLY these material/texture changes:

- Medium-height steel shelving units: grey/silver painted steel with bolt connections, shelf lips, and slight dents from use.
- Every box stays in its EXACT current position. Make them real cardboard — brown corrugated with shipping labels, barcodes, packing tape. Vary shade slightly (newer tan, older weathered brown).
- Fill empty shelf spaces with additional boxes, plastic totes, and shrink-wrapped bundles. Do NOT move existing items.
- Concrete floor with wear — grey with hairline cracks, tire scuffs, dust along shelf bases, faded yellow aisle markings.
- Overhead fluorescent tube lighting, harsh white light with visible fixtures.
- Arched roof becomes corrugated steel with metal trusses and rivets.
- Slight atmospheric dust haze where lights shine down.
- Person in the distance becomes a real warehouse worker in hi-vis vest.

Style: photorealistic industrial photography, overhead drone security camera, distribution center picking aisle. Output must be a texture upgrade of the input, NOT a new image."""

ENDCAP_PROMPT = """CRITICAL: This is a layout-preservation task. The warehouse structure, shelf positions, aisle width, camera angle, and every object's exact placement must remain IDENTICAL to the input image. Do NOT redesign, rearrange, or reinterpret the layout. Only change textures, materials, and lighting to make it photorealistic.

Transform this 3D warehouse render into a photorealistic photograph. The geometry and composition must be a 1:1 match to the input — same number of shelves, same aisle width, same camera height and angle, same box positions.

Make ONLY these material/texture changes:

- Medium-height steel shelving units: grey/silver painted steel with bolt connections, shelf lips, and slight dents from use.
- Every box stays in its EXACT current position. Make them real cardboard — brown corrugated with shipping labels, barcodes, packing tape. Vary shade slightly (newer tan, older weathered brown).
- Fill empty shelf spaces with additional boxes, plastic totes, and shrink-wrapped bundles. Do NOT move existing items.
- Any tall pallet racking visible: keep exact structure, upgrade textures to realistic painted steel with scratches and wear. Fill all levels with loaded skids — pallets with shrink-wrapped boxes and crates.
- Concrete floor with wear — grey with hairline cracks, tire scuffs, dust along shelf bases, faded yellow aisle markings.
- Overhead fluorescent tube lighting, harsh white light with visible fixtures.
- Arched roof becomes corrugated steel with metal trusses and rivets.
- Slight atmospheric dust haze where lights shine down.

Style: photorealistic industrial photography, overhead drone security camera, distribution center picking aisle. Output must be a texture upgrade of the input, NOT a new image."""

SHELVING_RANGES = [(21, 45), (66, 170)]
ENDCAP_RANGES = [(171, 195)]


def log(msg):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def get_aisle_type(frame_num):
    for s, e in ENDCAP_RANGES:
        if s <= frame_num <= e:
            return "endcap"
    for s, e in SHELVING_RANGES:
        if s <= frame_num <= e:
            return "shelving"
    return "forklift"


def get_frames_to_process(start, end):
    frames = []
    for f in sorted(glob.glob(os.path.join(BASE_DIR, "frame_*.png"))):
        num = int(Path(f).stem.split("_")[1])
        if start <= num <= end:
            out_path = os.path.join(OUTPUT_DIR, f"frame_{num:04d}.png")
            if not os.path.exists(out_path):
                frames.append(num)
    return frames


def process_frame(page, frame_num, retry=0):
    base_path = os.path.join(BASE_DIR, f"frame_{frame_num:04d}.png")
    save_path = os.path.join(OUTPUT_DIR, f"frame_{frame_num:04d}.png")
    aisle_type = get_aisle_type(frame_num)
    prompt = {"shelving": SHELVING_PROMPT, "endcap": ENDCAP_PROMPT, "forklift": FORKLIFT_PROMPT}[aisle_type]

    try:
        # ==== STEP 1: New chat ====
        log(f"  [1/7] New chat...")
        page.goto("https://gemini.google.com/app")
        page.wait_for_load_state("load", timeout=20000)
        time.sleep(3)

        # ==== STEP 2: Click "Tools" then "Create image" ====
        log(f"  [2/7] Tools → Create image...")
        tools_btn = page.get_by_text("Tools", exact=True).first
        tools_btn.wait_for(state="visible", timeout=5000)
        tools_btn.click()
        time.sleep(1)

        create_img = page.get_by_text("Create image", exact=True).first
        create_img.wait_for(state="visible", timeout=3000)
        create_img.click()
        time.sleep(1)

        # ==== STEP 3: Click "+" then "Upload files" ====
        log(f"  [3/7] Uploading frame_{frame_num:04d}.png ({aisle_type})...")
        # The "+" button = "Open upload file menu"
        plus_btn = page.locator('button[aria-label="Open upload file menu"]').first
        plus_btn.wait_for(state="visible", timeout=5000)
        plus_btn.click()
        time.sleep(1)

        # "Upload files" from menu
        with page.expect_file_chooser(timeout=5000) as fc_info:
            page.get_by_text("Upload files", exact=True).first.click()
        fc_info.value.set_files(base_path)

        # Wait for image to fully upload and show preview
        log(f"    Waiting for image to attach...")
        time.sleep(5)

        # Verify image is attached (look for image preview thumbnail)
        for check in range(10):
            # Check for uploaded image preview in input area
            previews = page.locator('img[src^="blob:"], .upload-preview, .file-preview').all()
            if previews:
                log(f"    Image attached!")
                break
            time.sleep(1)
        else:
            log(f"    WARNING: Could not confirm image attachment, continuing anyway...")
        time.sleep(1)

        # ==== STEP 4: Paste prompt ====
        log(f"  [4/7] Pasting prompt...")
        input_field = page.locator('div[contenteditable="true"]').first
        input_field.wait_for(state="visible", timeout=5000)
        input_field.click()
        time.sleep(0.5)

        # Use clipboard to paste (fast)
        page.evaluate("(text) => navigator.clipboard.writeText(text)", prompt)
        time.sleep(0.3)
        page.keyboard.press("Meta+v")
        time.sleep(1.5)

        # Verify paste worked
        content = input_field.inner_text()
        if len(content) < 50:
            log(f"    Clipboard paste failed, typing directly...")
            input_field.click()
            page.keyboard.press("Meta+a")
            page.keyboard.press("Backspace")
            page.keyboard.type(prompt, delay=1)
            time.sleep(1)

        # ==== STEP 5: Send ====
        log(f"  [5/7] Sending...")
        sent = False
        for sel in ['button[aria-label="Send message"]', 'button[aria-label="Submit"]',
                    'button[data-tooltip="Send"]']:
            try:
                btn = page.locator(sel).first
                if btn.is_visible(timeout=1500):
                    btn.click()
                    sent = True
                    break
            except Exception:
                continue
        if not sent:
            page.keyboard.press("Enter")
        time.sleep(3)

        # ==== STEP 6: Wait for generation ====
        log(f"  [6/7] Waiting for generation...")
        image_ready = False

        for tick in range(60):  # 180s max
            time.sleep(3)
            elapsed = (tick + 1) * 3

            # Generated images use lh3.googleusercontent.com URLs with "AI generated" alt
            ai_imgs = page.locator('img[alt*="AI generated"]').all()

            # Also check for "Good response" button (appears when done)
            thumbs = page.locator('button[aria-label="Good response"]').all()

            if ai_imgs and thumbs:
                image_ready = True
                log(f"    Generated! ({elapsed}s)")
                time.sleep(3)
                break

            if ai_imgs and elapsed > 20:
                image_ready = True
                log(f"    Image found ({elapsed}s)")
                time.sleep(5)
                break

            # Check if response stopped (error)
            stopped = page.get_by_text("Response stopped").all()
            if stopped:
                log(f"    Gemini stopped response at {elapsed}s!")
                break

            if elapsed % 15 == 0:
                log(f"    Waiting... ({elapsed}s)")

        if not image_ready:
            log(f"    TIMEOUT after 180s")
            return False

        # ==== STEP 7: Download the generated image ====
        log(f"  [7/7] Downloading...")
        time.sleep(2)

        # Method 1: Hover image to show overlay, click download button
        try:
            ai_img = page.locator('img[alt*="AI generated"]').first
            ai_img.scroll_into_view_if_needed(timeout=5000)
            time.sleep(1)
            ai_img.hover()
            time.sleep(1)

            dl_btn = page.locator('button[data-test-id="download-generated-image-button"]').first
            with page.expect_download(timeout=15000) as dl_info:
                dl_btn.click(timeout=5000)
            download = dl_info.value
            download.save_as(save_path)
            file_size = os.path.getsize(save_path)
            log(f"    SAVED (overlay): frame_{frame_num:04d}.png ({file_size // 1024}KB)")
            return True
        except Exception as e:
            log(f"    Overlay download failed: {e}")

        # Method 2: Click "..." menu then "Download image"
        try:
            dots_btn = page.locator('button[data-test-id="more-menu-button"]').first
            dots_btn.scroll_into_view_if_needed(timeout=3000)
            dots_btn.click(timeout=3000)
            time.sleep(1)

            with page.expect_download(timeout=15000) as dl_info:
                page.get_by_text("Download image", exact=True).first.click(timeout=3000)
            download = dl_info.value
            download.save_as(save_path)
            file_size = os.path.getsize(save_path)
            log(f"    SAVED (menu): frame_{frame_num:04d}.png ({file_size // 1024}KB)")
            return True
        except Exception as e:
            log(f"    Menu download failed: {e}")

        return False

    except Exception as e:
        log(f"  ERROR: {e}")
        if retry < 2:
            log(f"  Retrying (attempt {retry + 2})...")
            time.sleep(5)
            return process_frame(page, frame_num, retry + 1)
        return False


def run_automation(start, end, delay=15, overnight=False):
    frames = get_frames_to_process(start, end)
    if not frames:
        print("All frames already processed!")
        return

    with open(LOG_FILE, "w") as f:
        f.write(f"=== Started {datetime.datetime.now()} ===\n")
        f.write(f"Frames: {len(frames)} ({frames[0]} to {frames[-1]})\n\n")

    log(f"{'='*60}")
    log(f"  GEMINI AUTOMATION")
    log(f"  {len(frames)} frames: {frames[0]} to {frames[-1]}")
    log(f"  Mode: {'OVERNIGHT' if overnight else 'INTERACTIVE'}")
    log(f"{'='*60}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with sync_playwright() as p:
        # Connect to your already-running Chrome
        log("Connecting to Chrome on port 9222...")
        try:
            browser = p.chromium.connect_over_cdp("http://localhost:9222")
        except Exception as e:
            print(f"\nERROR: Could not connect to Chrome: {e}")
            print("\nMake sure you did these steps:")
            print("  1. Quit Chrome completely (Cmd+Q)")
            print("  2. Run in terminal:")
            print('     /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome --remote-debugging-port=9222')
            print("  3. Then run this script in another terminal tab")
            return

        context = browser.contexts[0]
        page = context.new_page()
        page.goto("https://gemini.google.com/app")
        time.sleep(3)

        log("Connected! Checking if logged in...")

        # Check if signed in (look for "Sign in" button = not logged in)
        sign_in = page.get_by_text("Sign in", exact=True).all()
        if sign_in:
            print("\n  >>> You need to sign into Gemini in Chrome first!")
            print("  >>> Go to the Chrome window and log in, then press ENTER here...")
            input()

        print("\n  >>> Ready! Press ENTER to start processing...\n")
        input()

        success = 0
        failed = 0
        failed_frames = []

        for i, frame_num in enumerate(frames):
            log(f"\n[{i+1}/{len(frames)}] Frame {frame_num:04d} "
                f"({get_aisle_type(frame_num)})")

            if process_frame(page, frame_num):
                success += 1
            else:
                failed += 1
                failed_frames.append(frame_num)
                if not overnight:
                    log("  >> FAILED. Press ENTER to continue...")
                    input()
                else:
                    log("  >> SKIPPED (overnight mode)")

            if (i + 1) % 10 == 0:
                log(f"\n  --- PROGRESS: {success} ok, {failed} failed, "
                    f"{len(frames) - i - 1} remaining ---")

            if i < len(frames) - 1:
                time.sleep(delay)

        log(f"\n{'='*60}")
        log(f"  DONE! Success: {success} | Failed: {failed}")
        if failed_frames:
            log(f"  Failed frames: {failed_frames}")
        log(f"{'='*60}")

        page.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=73)
    parser.add_argument("--end", type=int, default=195)
    parser.add_argument("--delay", type=int, default=15)
    parser.add_argument("--overnight", action="store_true")
    args = parser.parse_args()

    run_automation(args.start, args.end, args.delay, args.overnight)
