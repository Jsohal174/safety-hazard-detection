"""
Inject safety hazards into photorealistic warehouse images via Gemini.
Uses the same Playwright + Chrome CDP automation as gemini_automate.py.

Prompt diversity: Each hazard prompt is assembled from randomized slot pools,
giving 15,000-25,000+ unique combinations per category.

SETUP (same as gemini_automate.py):
  1. Quit Chrome completely (Cmd+Q)
  2. Run: /Applications/Google Chrome.app/Contents/MacOS/Google Chrome --remote-debugging-port=9222
  3. Log into Gemini if needed.
  4. Then run this script in another terminal tab.

Usage:
    python scripts/gemini_hazard_inject.py --category spill --start 0 --end 170 --overnight
    python scripts/gemini_hazard_inject.py --category missing_ppe --start 0 --end 170 --overnight
    python scripts/gemini_hazard_inject.py --category improper_stacking --start 0 --end 170 --overnight
    python scripts/gemini_hazard_inject.py --category forklift_violation --start 0 --end 20 --variants 3 --overnight
"""

import argparse
import json
import glob
import os
import random
import time
import datetime
from pathlib import Path
from playwright.sync_api import sync_playwright

INPUT_DIR = "/Users/jaskiratsinghsohal/Desktop/safety-hazard-detection/realistic"
OUTPUT_BASE = "/Users/jaskiratsinghsohal/Desktop/safety-hazard-detection/outputs/datasets/images"
LABELS_FILE = "/Users/jaskiratsinghsohal/Desktop/safety-hazard-detection/outputs/datasets/labels.json"
LOG_FILE = "/Users/jaskiratsinghsohal/Desktop/safety-hazard-detection/hazard_injection.log"

# ============================================================
# PROMPT WRAPPER
# ============================================================

WRAPPER = """CRITICAL: Keep this photorealistic warehouse photograph EXACTLY as it is — same camera angle, same shelving, same floor, same lighting, same perspective. Do NOT redesign or rearrange anything. The warehouse layout, structure, and every existing object must remain IDENTICAL.

Only add the following hazard to the existing scene:

{hazard}

Everything else in the image — shelving, boxes, floor, walls, lighting — must remain EXACTLY as in the input. The output should look like the same photograph with the hazard naturally present. Do NOT change the camera angle, perspective, or warehouse layout."""

STACKING_FORKLIFT_WRAPPER = """CRITICAL: Keep this photorealistic warehouse photograph EXACTLY as it is — same camera angle, same floor, same lighting, same perspective, same warehouse layout. Do NOT redesign, rearrange, or restructure anything.

Pick ONE existing pallet already sitting flat on the racking and modify ONLY the boxes and wrap on that pallet to show a stacking violation. The pallet/skid itself MUST stay flat and level on the rack beams — do NOT tilt, lean, or angle the pallet. Do NOT show anything falling, sliding off, or hanging over the edge. The skid stays exactly where it is on the beams.

{hazard}

IMPORTANT: The pallet base stays FLAT on the rack beams. Only the boxes on top of it look wrong — messy, crushed, shifted, torn wrap, broken straps, mixed sizes. Nothing is falling or about to fall. The result should look like a normal warehouse photo where one pallet was just loaded carelessly."""

STACKING_SHELVING_WRAPPER = """CRITICAL: Keep this photorealistic warehouse photograph EXACTLY as it is — same camera angle, same floor, same lighting, same perspective, same warehouse layout.

Do NOT add any new objects, shelving units, pallets, skids, or metal structures. Do NOT place boxes on the ground or in the aisle. Do NOT show boxes floating or falling in the air.

ONLY change: make some of the existing cardboard boxes that are already on the existing steel shelves look improperly stacked. The boxes stay on the same shelves they are already on — just make them look messy and unsafe:

{hazard}

The shelves, floor, walls, ceiling, and everything else stays EXACTLY the same. Only the arrangement of boxes already on the shelves changes."""

# ============================================================
# SPILL SLOTS — Based on real OSHA warehouse spill data
# Real spills are dark/clear/muted, not bright colors.
# Key visual cue from drone: reflective sheen on concrete.
# ============================================================

SPILL_TEMPLATE = "A {size} {liquid} on the concrete floor {position}, {shape}. The spill appears {surface}. {context}"

SPILL_SLOTS = {
    "size": [
        "small (about 30cm)", "medium (about 1 meter wide)", "large (about 1.5 meter wide)",
        "thin but long (about 2 meters)", "small cluster of drips spread over half a meter",
    ],
    "liquid": [
        "dark amber hydraulic oil puddle", "clear water puddle from a ceiling leak",
        "dark brown motor oil stain", "slightly tinted condensation puddle",
        "dark wet patch from a forklift fluid leak", "murky grey-brown dirty water puddle",
        "dark oily residue patch", "clear water pooling from a pipe drip",
        "amber-brown forklift hydraulic fluid spill", "dark greasy wet spot",
    ],
    "position": [
        "in the center of the aisle", "near the base of the left shelving",
        "near the right shelving uprights", "in the foreground of the aisle",
        "at the far end of the aisle", "pooling in a low spot in the concrete",
        "at the edge of the aisle near a shelf upright", "between the two rows of shelving",
    ],
    "shape": [
        "forming an irregular puddle with uneven edges",
        "as a long smeared streak across the floor",
        "as an organic-shaped pool following the floor slope",
        "as a thin wet film across the concrete",
        "as a concentrated dark patch with drip trails leading away from it",
        "pooling in the concrete floor joints and cracks",
        "as scattered drips and a main puddle",
    ],
    "surface": [
        "glossy and reflective, catching the overhead warehouse lights",
        "dark and wet against the lighter grey concrete",
        "with a slight oily sheen reflecting fluorescent light",
        "as a wet darkened patch on the concrete, clearly visible from above",
        "with a dull matte surface from thick viscous fluid",
        "fresh and wet with sharp edges where liquid meets dry concrete",
    ],
    "context": [
        "with no wet floor sign or warning nearby",
        "with faint forklift tire tracks running through it",
        "near a cardboard box on the floor that appears to be leaking",
        "with a drip trail leading back toward the shelving",
        "near some grey absorbent granules sprinkled at the edge but not covering it",
        "with wet boot prints visible walking through the puddle",
        "next to a tipped-over plastic container on its side",
        "appearing to drip down from a shelf above, with drip marks on the shelf upright",
    ],
}

# ============================================================
# MISSING PPE SLOTS — Based on OSHA warehouse PPE requirements
# From drone (3.5m, 55 deg): hard hat = colored dome on head,
# hi-vis vest = bright yellow/green torso. Missing = dark hair,
# dark/muted clothing visible from above.
# ============================================================

MISSING_PPE_TEMPLATE = "A {appearance} is {action}, positioned {position}. From this overhead angle, {head_detail}. {torso_detail}. {context}"

MISSING_PPE_SLOTS = {
    "appearance": [
        "warehouse worker in dark jeans and a grey t-shirt",
        "person in a black hoodie and dark cargo pants",
        "worker in a navy blue long-sleeve shirt and work pants",
        "person in a dark green jacket and jeans",
        "worker in plain brown coveralls with no reflective strips",
        "person in a dark red flannel shirt and jeans",
        "worker in all-black clothing, blending into the shadows",
        "person in a grey sweatshirt and dark pants",
    ],
    "action": [
        "walking down the aisle", "reaching up toward a shelf",
        "crouching near the floor picking something up", "carrying a cardboard box",
        "pushing a hand cart", "standing still looking at a clipboard",
        "pulling a pallet jack", "bending over organizing items on a low shelf",
    ],
    "position": [
        "in the center of the aisle", "in the mid-ground near the left shelving",
        "in the background of the aisle", "in the foreground near the right shelf",
        "at the end of the aisle", "between two shelf sections",
        "near an aisle intersection", "next to a shelf upright",
    ],
    "head_detail": [
        "their dark hair is clearly visible on top of their head with no hard hat",
        "you can see the top of their bare head with no safety helmet",
        "they have a baseball cap on instead of a proper hard hat",
        "their head shows no hard hat, just dark hair from above",
        "no safety helmet visible on their head, only hair",
        "they are wearing a beanie/toque instead of a required hard hat",
    ],
    "torso_detail": [
        "Their torso is dark-colored with no high-visibility vest at all",
        "They have no reflective or hi-vis clothing, just plain dark fabric",
        "No fluorescent yellow or orange vest is visible on their body",
        "Their clothing is entirely muted and dark, with zero high-visibility gear",
        "They are wearing no reflective vest, completely blending into the dim warehouse",
    ],
    "context": [
        "A PPE Required sign is visible on a nearby shelf upright",
        "Another worker in the distance is wearing a proper yellow hi-vis vest and white hard hat for contrast",
        "They are in a forklift traffic zone where PPE is mandatory",
        "Shelving above them has heavy items, making a hard hat essential",
        "They are working alone in the aisle with no supervisor visible",
        "Yellow floor markings indicate this is a mandatory PPE zone",
    ],
}

# ============================================================
# FORKLIFT VIOLATION SLOTS — Research-backed, OSHA data
#
# 10 violation types visible from drone perspective:
#   1. Elevated forks while traveling (#1 OSHA citation)
#   2. Overloaded / capacity exceeded
#   3. Pedestrian in danger zone (20% of accidents, 36% of deaths)
#   4. Tipover risk (24% of accidents, 42% of injuries)
#   5. Person riding on forks (illegal)
#   6. Falling / sliding load
#   7. Unattended with forks up
#   8. Blocking aisle or exit
#   9. Pedestrian pinned against racking (25% of injuries)
#  10. Load blocking operator visibility
#
# Template: Each prompt picks ONE primary violation + secondary
# details for maximum realism. ~50,000+ unique combinations.
# ============================================================

FORKLIFT_TEMPLATE = "The forklift is {forklift_position}, {forklift_orientation}. A {operator_detail} is operating it. {violation}. {secondary_detail}. {environment}."

FORKLIFT_SLOTS = {
    "forklift_position": [
        "in the center of the aisle",
        "in the foreground close to the camera, taking up a large portion of the frame",
        "in the far background of the aisle, smaller but clearly visible",
        "near the left-side racking, close to the shelf uprights",
        "near the right-side racking, almost touching the pallet rack",
        "at the far end of the aisle approaching an intersection",
        "in the mid-ground of the aisle, roughly halfway down",
        "at the near end of the aisle closest to the camera",
        "slightly off-center toward the left side of the aisle",
        "slightly off-center toward the right side of the aisle",
    ],
    "forklift_orientation": [
        "driving toward the camera (front of forklift facing the viewer)",
        "driving away from the camera (rear of forklift facing the viewer)",
        "turning left with the front wheels angled sharply",
        "turning right with the front wheels angled sharply",
        "parked at a diagonal angle across the aisle",
        "reversing with the operator looking over their shoulder",
        "moving perpendicular to the aisle, crossing between rack rows",
        "driving forward at speed down the center of the aisle",
        "stopped mid-aisle with the operator reaching for something on a shelf",
        "backing up with the load leading the way",
    ],
    "operator_detail": [
        "male worker wearing a hard hat and hi-vis yellow vest",
        "worker in a blue uniform shirt and safety glasses",
        "young worker in orange hi-vis coveralls",
        "worker wearing a hard hat but no seatbelt — the belt hangs loose",
        "worker in a grey t-shirt with no hard hat or safety vest at all",
        "female operator in standard warehouse PPE (hard hat, vest, boots)",
        "distracted operator looking down at a clipboard instead of ahead",
        "worker with one hand on the wheel and the other holding a phone to their ear",
        "worker in a winter jacket who looks like they're rushing through a task",
        "operator leaning out of the cab to see around the load",
    ],
    "violation": [
        # --- 1. ELEVATED FORKS WHILE TRAVELING (most cited) ---
        "The forklift is driving down the aisle with its forks raised about 2 meters off the ground — they should be lowered to 10-15cm when traveling. The elevated steel forks are clearly visible from above extending forward at chest height",
        "The forklift is traveling with the mast fully extended and forks raised to the third rack level while still moving forward, creating a massive tip-over risk. From above the raised forks cast a long shadow on the floor",
        "The forklift is moving through the aisle with empty forks elevated waist-high instead of lowered to the ground. The forks stick out prominently from the front of the machine visible from the overhead angle",
        "The forklift is driving with forks raised to about 1.5 meters, carrying nothing — the operator forgot to lower them after the last pick. The bare metal forks are clearly extended above ground level",

        # --- 2. OVERLOADED / DOUBLE-STACKED ---
        "The forklift is carrying two pallets stacked on top of each other on the forks — double the rated capacity. The top pallet is wobbling and shifting. From above, the oversized double-high load dwarfs the forklift",
        "The forklift has an enormous single load that extends well past the fork tips on both sides — the load is wider than the forklift itself. The rear wheels are barely touching the ground as the front dips under the weight",
        "The forklift is carrying a pallet stacked absurdly high with boxes — the stack is taller than the mast and sways with every movement. Seen from above the tall unstable tower of boxes looks ready to topple",
        "The forklift is hauling a pallet of heavy steel drums that clearly exceeds the weight capacity — the front tires are compressed flat and the rear of the forklift is lifting slightly off the ground",

        # --- 3. PEDESTRIAN IN DANGER ZONE ---
        "A warehouse worker in a hi-vis vest is walking directly in the forklift's travel path just 3 meters ahead, looking at their phone and completely unaware of the approaching forklift behind them",
        "A pedestrian is stepping out from behind the end of a shelf rack directly into the forklift's blind spot. The forklift is moving toward the intersection and neither can see the other",
        "Two workers are standing in the middle of the aisle having a conversation while the forklift approaches from behind them. They are directly in the travel lane with no awareness",
        "A worker is crouched down on the floor picking up a dropped item directly in the forklift's path. The operator's view is blocked by the load and cannot see the person on the ground ahead",
        "A pedestrian is walking alongside the forklift within arm's reach — inside the crush zone between the forklift and the shelf racking. One sudden turn would pin them against the metal uprights",

        # --- 4. TIPOVER RISK ---
        "The forklift is making a sharp turn at the end of the aisle with a heavy raised load, causing the entire machine to tilt visibly to one side. Two wheels on the inside of the turn are barely touching or lifting off the ground",
        "The forklift is driving diagonally across the aisle at speed with an elevated load, leaning dangerously to one side as it turns. The load is shifting sideways on the forks from the centrifugal force",
        "The forklift is driving over an uneven patch of floor with one side of the machine notably higher than the other, causing the load to lean. The tilt is clearly visible from the overhead drone angle",

        # --- 5. PERSON RIDING ON FORKS ---
        "A second worker is standing on the raised forks of the forklift being lifted up to reach a high shelf — using the forklift as an improvised man-lift. The person is balancing on the flat forks with no fall protection, several meters off the ground",
        "A worker is riding on the outside of the forklift, standing on the counterweight at the back and holding onto the overhead guard while the forklift moves. From above you can see the unauthorized rider clinging to the rear of the machine",
        "Two people are on the forklift — the operator in the seat and a second person sitting on the forks riding along for a lift down the aisle. The passenger has their legs dangling off the side of the forks",

        # --- 6. FALLING / SLIDING LOAD ---
        "Boxes are actively sliding off the forklift's raised pallet — two boxes have already fallen onto the floor behind the forklift and more are teetering on the edge about to fall. Loose cardboard boxes are scattered on the concrete",
        "The load on the forklift's forks is completely unwrapped with no shrink wrap or strapping, and the top boxes have shifted to one side, about to slide off. One box is hanging half off the edge of the pallet",
        "A pallet on the forklift's forks has broken — one side of the wooden skid has snapped and boxes are spilling through the gap. Several boxes have already fallen to the floor below the forks creating debris in the aisle",
        "The forklift is carrying a tall stack of boxes with the forks tilted forward instead of back, and the entire load is sliding forward toward the tips of the forks. The front boxes are hanging past the fork ends",

        # --- 7. UNATTENDED WITH FORKS UP ---
        "The forklift is parked in the middle of the aisle with no operator in the seat, engine still running, and forks raised about 1 meter off the ground. The empty operator seat is visible from above. OSHA requires forks lowered and engine off when unattended",
        "The forklift sits abandoned in the aisle with forks elevated to the second rack level still holding a pallet — the operator walked away mid-task. The raised load blocks half the aisle and the empty cab is visible from above",
        "An unattended forklift is parked at the end of the aisle with forks raised high and no operator anywhere nearby. The key is still in the ignition visible from the overhead angle. Anyone could walk under the elevated forks",

        # --- 8. BLOCKING AISLE / EXIT ---
        "The forklift is parked sideways across the aisle completely blocking the pathway. No one can walk or drive through — the entire aisle width is blocked by the machine. A worker on the other side appears to be waiting to get past",
        "The forklift is parked diagonally in the aisle with a pallet half-placed on a rack, blocking 80% of the pathway. The remaining gap is too narrow for a person to safely walk through",
        "The forklift is stopped in front of an emergency exit door, completely blocking access to it. The exit sign is visible above and the forklift with its load is parked right against the door, preventing evacuation",

        # --- 9. PEDESTRIAN PINNED / CRUSH RISK ---
        "A worker is squeezed between the side of the forklift and the shelf racking — trapped in the narrow gap with the forklift pressing them against the metal upright. The forklift has moved too close to the rack while the person was between them",
        "A pedestrian is backed against the racking with the forklift approaching head-on in the narrow aisle — there is nowhere for them to go. The person is pressing themselves flat against the shelving as the forklift advances",

        # --- 10. LOAD BLOCKING VISIBILITY ---
        "The forklift is driving forward with a massive tall load on the forks that completely blocks the operator's forward view — they cannot see the aisle ahead at all. OSHA requires traveling in reverse with view-blocking loads, but this operator is going forward blind",
        "The forklift is carrying a wide load of stacked boxes that extends past both sides of the machine, blocking the operator's view to the left and right. The operator cannot see around the load to check for pedestrians at intersections",
    ],
    "secondary_detail": [
        # --- Additional hazard details ---
        "The pallet on the forks has no shrink wrap — loose boxes could shift and fall at any moment",
        "The load on the forks extends several inches past the fork tips, hanging over the ends",
        "Boxes on the forks are stacked well above the forklift's load backrest, completely unsecured",
        "The forklift's load is visibly off-center, shifted to one side of the forks creating imbalance",
        "The forks are carrying a single oversized awkward item that is not centered and could slide",
        "The concrete floor shows old oil stains and the forklift tires appear to be on a slightly slippery surface",
        "Debris from a previously dropped load — broken pallet wood and scattered items — is on the floor nearby",
        "The forklift's overhead guard has visible dent damage from a previous collision with racking",
        "A yellow speed limit sign reading 8 KM/H is visible on a nearby rack upright — the forklift appears to be exceeding it",
        "Tire marks on the concrete floor show the forklift made a sharp aggressive turn in this spot",
        "A 'CAUTION: FORKLIFT TRAFFIC' sign is posted on the end of the racking but partially obscured",
        "The forklift's warning beacon/strobe light on top is not illuminated despite being in an active traffic area",
        "No horn was sounded — there are no audible warnings and pedestrians ahead are unaware",
        "A convex safety mirror at the aisle intersection is present but the forklift operator is not checking it",
    ],
    "environment": [
        "The aisle is narrow with tall pallet racking close on both sides leaving minimal clearance",
        "This section of the warehouse is near an aisle intersection where cross-traffic is expected",
        "The warehouse floor has faded yellow pedestrian walkway lines that the forklift is crossing over",
        "Other workers are visible in the background going about their tasks in adjacent aisles",
        "The overhead fluorescent lights create harsh shadows making it harder to see ground-level hazards",
        "The area appears to be a high-traffic zone with multiple pallets staged along the aisle edges",
        "Nearby shelving is fully loaded with heavy inventory on every level increasing the stakes of a collision",
        "The warehouse is clearly during active operations — signs of a busy shift with movement and activity",
        "The concrete floor is smooth and slightly dusty, typical of a warehouse with heavy forklift traffic",
        "An emergency exit sign is visible at the end of the aisle in the background",
    ],
}

# ============================================================
# IMPROPER STACKING SLOTS — Based on OSHA 1910.176(b), 1917.14
# Research: Real violations are SUBTLE (3-10 deg lean). Key cues:
# overhanging edges, damaged skids, crushed bottom boxes, torn/
# missing shrink wrap, broken strapping, shifted loads, mixed sizes.
# Forklift aisle (tall pallet racking) + shelving aisle (medium racks).
# 6 dimensions each like spills for maximum variety.
# ============================================================

# --- FORKLIFT AISLE: skids on tall pallet racking ---
STACKING_FORKLIFT_TEMPLATE = "A skid {position} has {violation}. The wooden pallet {skid_condition}. The boxes {box_condition}. {securing}. {context}."

STACKING_FORKLIFT_SLOTS = {
    "position": [
        "on the left side racking, second beam level up",
        "on the right side racking, at the lowest beam level",
        "on the left side racking, on the top level near the ceiling",
        "on the right side racking, third level up",
        "on the left side racking, just above floor height on the first beam",
        "on the right side racking, at the highest rack level",
        "on the left side racking, at mid-height",
        "on the right side racking, second level from the top",
        "on the left side racking, on the second beam level near the middle of the aisle run",
        "on the right side racking, at the lowest level in the foreground",
    ],
    "violation": [
        "its boxes shifted off-center from rough forklift placement — the top layer has slid a few inches to one side while the pallet itself sits flat on the beams",
        "a top-heavy load — large heavy boxes stacked on top of smaller lighter ones, with the bottom cartons visibly crushed and compressed under the weight",
        "boxes stacked in columns with no interlocking — all vertical joints aligned straight up the stack with no brick-lay pattern",
        "mixed-size boxes crammed together haphazardly with visible gaps between them and no organized stacking pattern",
        "boxes stacked unevenly — one side of the stack is one or two boxes taller than the other, creating a stepped uneven top",
        "its shrink wrap completely torn away, leaving the boxes loose and unsecured on the skid sitting on the beams",
        "broken strapping — the plastic bands that held the load together have snapped and the boxes have shifted apart slightly",
        "boxes that are clearly too heavy for the bottom layer — the lower cartons are visibly crushed with bulging sides and dented corners",
        "boxes stacked past the normal fill height for that rack level — the stack is noticeably taller than the pallets on neighboring bays",
        "its load arranged with no pattern — boxes rotated at random angles, different sizes mixed, creating a messy disorganized top surface when viewed from above",
    ],
    "skid_condition": [
        "has a cracked stringer — one of the three support beams has a visible vertical split causing that side to sag under the load",
        "is grey and weathered with one deck board slightly bowed from age and moisture damage",
        "has a missing deck board on one side, leaving a dark gap where a box above sags into the void",
        "has visible forklift tine damage — splintered gouges along the stringer edges where forks scraped during rough handling",
        "is darkened and discolored from moisture with warped boards that no longer sit flat",
        "has a crushed corner from forklift impact, making the whole load lean toward that damaged corner",
        "has a broken front deck board hanging down — snapped when the forklift placed it too hard on the beams",
        "looks mostly intact but the wood is old with hairline cracks along the grain of the stringers",
        "has the bottom boards scraped and frayed from being dragged across the warehouse floor",
        "has nail heads poking up through the top deck boards where the fasteners have worked loose from vibration",
    ],
    "box_condition": [
        "show crushed corners and bulging cardboard sides on the bottom layer from bearing the full stack weight above",
        "have the top layer shifted a few inches to one side, clearly misaligned from the layers below — a fan-shaped shift",
        "are a mix of different sizes crammed onto one skid, with small boxes wedged between larger heavy ones",
        "have dented compressed edges with wrinkled shipping labels from being squeezed during rough stacking",
        "are overhanging the pallet edge by 3-4 inches on one side, with brown cardboard extending past the wooden skid",
        "are rotated at slightly different angles so the top surface viewed from above is not a clean rectangle",
        "have visible gaps opening between them where the load shifted — dark lines between boxes that should be touching",
        "include several with collapsed top flaps that have caved inward under the weight of boxes stacked on them",
        "are all facing the same direction in column stacking with visible straight vertical joint lines — no interlocking",
        "have the bottom boxes darker and compressed while the top ones are lighter and intact — clear compression damage",
    ],
    "securing": [
        "There is no shrink wrap at all — the boxes sit completely loose and unsecured on the skid",
        "The shrink wrap has torn open on one side with a visible rip in the clear plastic, and boxes are shifting through the gap",
        "Only the bottom half of the pallet is wrapped — the top boxes above the wrap line are completely exposed and loose",
        "A broken plastic strap hangs off the side of the pallet, curled back on itself where it snapped — the load is no longer secured",
        "There is no banding or wrap whatsoever — individual cartons just sit on the skid with nothing holding them together",
        "Remnants of torn stretch wrap hang off one side like ragged clear plastic strips, no longer securing anything",
        "The wrap only covers the middle section — triangular dog-ear flaps of excess film poke out at the corners",
        "Two snapped blue plastic straps dangle from the pallet sides, the buckle clips still attached to one end of each",
        "The stretch wrap is so thin and overstretched it looks nearly transparent with a whitened frosted appearance — it provides no real support",
        "There are compression marks on the boxes where strapping used to be, but the straps are completely gone now",
    ],
    "context": [
        "The adjacent pallets on the same rack level are neatly stacked and wrapped for contrast",
        "This skid clearly stands out from the orderly pallets around it on the racking",
        "The overhang or lean is visible compared to the straight beam lines of the rack",
        "From the overhead drone camera, the misalignment of this pallet is obvious against the grid of the racking",
        "The damaged or shifted load contrasts with the properly stored pallets in the neighboring bays",
        "Looking down from the drone angle, gaps and shifted boxes on this pallet are clearly visible",
        "The pallet sticks out further into the aisle than the others on the same rack level",
        "From above, the uneven top surface of this stack contrasts with the flat tops of adjacent pallets",
    ],
}

# --- SHELVING AISLE: existing boxes on steel wire shelves made to look messy ---
STACKING_SHELVING_TEMPLATE = "Make the existing boxes {position} look {violation}. {detail}."

STACKING_SHELVING_SLOTS = {
    "position": [
        "on the left shelving, second level up",
        "on the right shelving, bottom level",
        "on the left shelving, top shelf",
        "on the right shelving, third level",
        "on the left shelving, lowest shelf",
        "on the right shelving, highest shelf",
        "on the left shelving, mid-height",
        "on the right shelving, second from top",
        "on the left shelving in the foreground",
        "on the right shelving near the far end",
    ],
    "violation": [
        "tilted and crooked — one box angled on top of another, not sitting flat",
        "crushed and compressed — the bottom box is dented and buckling under the weight of the boxes stacked on it",
        "messy and disorganized — boxes rotated at different angles, not aligned with each other",
        "overloaded — too many boxes crammed onto the shelf, some being squeezed outward",
        "top-heavy — a large heavy box sitting on top of a smaller lighter box that is being crushed",
        "unevenly stacked — some spots two boxes high, others three, creating a stepped uneven top",
        "shifted and gapped — boxes have slid apart leaving visible dark gaps between them",
        "turned the wrong way — boxes with FRAGILE and THIS SIDE UP labels are upside down or sideways",
        "piled carelessly — boxes thrown on the shelf at random angles like someone was in a rush",
        "damaged and dented — several boxes have crushed corners, torn flaps, and buckled cardboard sides",
    ],
    "detail": [
        "The messy boxes contrast with the neatly organized boxes on the adjacent shelves",
        "The uneven box arrangement is clearly visible from the overhead drone camera",
        "The bottom boxes are darker and compressed while the top ones still look intact",
        "Some box flaps are torn open from the pressure of boxes stacked on them",
        "The disorganized section stands out from the tidy rows on surrounding shelves",
        "Visible gaps and shadows between the shifted boxes make the mess obvious from above",
        "The boxes look like they were loaded hastily without care for proper stacking",
        "One box is noticeably crooked compared to the straight aligned boxes around it",
    ],
}

# ============================================================
# OBSTACLE SLOTS — Aisle obstructions creating trip/block hazards
# Based on OSHA 1910.176 (handling/storage) & 1910.22 (walking surfaces)
# Two variants: forklift aisle (wide, open) and shelving aisle (narrow)
# ============================================================

# --- FORKLIFT AISLE (frames 0-20): wide open area with forklift ---
OBSTACLE_FORKLIFT_TEMPLATE = "{object} is {position}. {severity}. {context}."

OBSTACLE_FORKLIFT_SLOTS = {
    "object": [
        "A manual pallet jack left abandoned on the warehouse floor",
        "A hand truck / dolly tipped over on its side",
        "A wooden pallet lying flat on the ground in the driving lane",
        "A broken wooden pallet with loose boards scattered around it",
        "A few cardboard boxes that fell off a pallet onto the floor",
        "Several plastic bins and a crate sitting on the warehouse floor",
    ],
    "position": [
        "in the middle of the open warehouse floor blocking the forklift driving path",
        "on the floor between the shelving rows and the forklift",
        "on the concrete floor in the main traffic area",
        "in the forklift travel lane on the warehouse floor",
    ],
    "severity": [
        "The obstruction blocks the forklift driving path",
        "The items create a trip hazard for pedestrians in the area",
        "The debris is spread over a 2-meter area on the floor",
    ],
    "context": [
        "The surrounding warehouse floor is otherwise clean",
        "No warning signs or cones are placed around the hazard",
        "The overhead lights illuminate the items clearly against the grey concrete",
    ],
}

OBSTACLE_FORKLIFT_WRAPPER = """CRITICAL: Keep this photorealistic warehouse photograph EXACTLY as it is — same camera angle, same shelving, same floor, same lighting, same perspective. Do NOT redesign or rearrange anything.

Only add the following obstruction to the existing scene on the warehouse floor:

{hazard}

The added items must be realistically sized (normal warehouse scale) and sitting on the concrete floor. The output must look like the SAME photograph with the obstruction naturally present."""

# --- SHELVING AISLE (frames 21-170): narrow aisles between pallet racking ---
OBSTACLE_SHELVING_TEMPLATE = "{object} is {position}. {severity}. {context}."

OBSTACLE_SHELVING_SLOTS = {
    "object": [
        "A single cardboard box lying on the aisle floor — it fell from a shelf",
        "Two or three cardboard boxes fallen on the narrow aisle floor",
        "Several small cardboard boxes scattered on the floor between the shelving rows",
        "A broken wooden pallet with loose boards on the aisle floor",
        "Pieces of broken wood from a crushed pallet on the aisle floor",
        "A few plastic storage bins that fell off a shelf onto the aisle floor",
        "A plastic crate and some loose items on the aisle floor",
        "A hand truck / dolly tipped over on its side in the narrow aisle",
        "A manual pallet jack left abandoned in the narrow aisle",
    ],
    "position": [
        "in the center of the narrow aisle blocking the walking path",
        "partially blocking the narrow aisle between the shelving rows",
        "in the middle of the aisle floor between the tall shelving racks",
        "near the base of the shelving, sticking out into the narrow aisle",
        "in the foreground of the narrow aisle close to the camera",
    ],
    "severity": [
        "The obstruction blocks more than half the narrow aisle width",
        "The items create a serious trip hazard in the tight aisle",
        "The debris is scattered across the narrow aisle floor",
        "The aisle is too narrow to safely step around the obstruction",
    ],
    "context": [
        "The rest of the aisle beyond the obstruction is clear and clean",
        "No warning signs or cones are placed around the hazard",
        "The overhead lights illuminate the items clearly against the grey concrete",
        "A nearby shelf has a visible gap where the fallen items came from",
    ],
}

OBSTACLE_SHELVING_WRAPPER = """CRITICAL: Keep this photorealistic warehouse photograph EXACTLY as it is — same camera angle, same shelving, same floor, same lighting, same perspective. Do NOT redesign or rearrange anything.

Only add the following obstruction to the existing aisle floor:

{hazard}

The added items must be realistically sized (normal warehouse scale) and sitting on the concrete floor between the shelving rows. The output must look like the SAME photograph with the obstruction naturally present."""

# ============================================================
# FORKLIFT VIOLATION WRAPPER — specific to forklift scenes
# ============================================================

FORKLIFT_WRAPPER = """CRITICAL: Keep this photorealistic warehouse photograph EXACTLY as it is — same camera angle, same floor, same lighting, same perspective. The warehouse layout, racking, and structure must remain IDENTICAL.

REPOSITION the forklift and ADD a human operator as described below. The forklift may need to be moved from its current location, resized, or reoriented — that is expected. Replace or modify the existing forklift to match:

{hazard}

REQUIREMENTS:
- The forklift must look like a real warehouse forklift (yellow/orange body, wear marks, industrial tires, mast and forks)
- The human operator must be clearly visible sitting in or on the forklift (seen from the overhead drone angle)
- The forklift should be the right size relative to the shelving and aisle width
- The violation must be clearly visible from the overhead drone camera angle
- Everything else in the warehouse (racking, boxes, floor, ceiling, lighting) stays exactly the same"""

# ============================================================
# CATEGORY REGISTRY
# ============================================================

CATEGORIES = {
    "spill": (SPILL_TEMPLATE, SPILL_SLOTS),
    "missing_ppe": (MISSING_PPE_TEMPLATE, MISSING_PPE_SLOTS),
    "forklift_violation": (FORKLIFT_TEMPLATE, FORKLIFT_SLOTS),
    "obstacle": None,  # handled specially — depends on frame number (forklift vs shelving aisle)
    "improper_stacking": None,  # handled specially — depends on frame number
}

# Forklift aisle = frames 0-20, shelving aisle = frames 21+
FORKLIFT_AISLE_END = 20


def log(msg):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def assemble_prompt(category, frame_num=0):
    """Randomly assemble a hazard prompt from slot pools."""
    if category == "improper_stacking":
        if frame_num <= FORKLIFT_AISLE_END:
            template = STACKING_FORKLIFT_TEMPLATE
            slots = STACKING_FORKLIFT_SLOTS
        else:
            template = STACKING_SHELVING_TEMPLATE
            slots = STACKING_SHELVING_SLOTS
    elif category == "obstacle":
        if frame_num <= FORKLIFT_AISLE_END:
            template = OBSTACLE_FORKLIFT_TEMPLATE
            slots = OBSTACLE_FORKLIFT_SLOTS
        else:
            template = OBSTACLE_SHELVING_TEMPLATE
            slots = OBSTACLE_SHELVING_SLOTS
    else:
        template, slots = CATEGORIES[category]
    filled = {k: random.choice(v) for k, v in slots.items()}
    hazard_text = template.format(**filled)
    if category == "improper_stacking":
        wrapper = STACKING_FORKLIFT_WRAPPER if frame_num <= FORKLIFT_AISLE_END else STACKING_SHELVING_WRAPPER
    elif category == "forklift_violation":
        wrapper = FORKLIFT_WRAPPER
    elif category == "obstacle":
        wrapper = OBSTACLE_FORKLIFT_WRAPPER if frame_num <= FORKLIFT_AISLE_END else OBSTACLE_SHELVING_WRAPPER
    else:
        wrapper = WRAPPER
    return wrapper.format(hazard=hazard_text), hazard_text


def get_frames_to_process(start, end, step, variant, category):
    """Get list of (frame_num, variant_idx) to process."""
    output_dir = os.path.join(OUTPUT_BASE, category)
    frames = []

    available = []
    for f in sorted(glob.glob(os.path.join(INPUT_DIR, "frame_*.png"))):
        num = int(Path(f).stem.split("_")[1])
        if start <= num <= end:
            available.append(num)

    selected = available[::step] if step > 1 else available

    for num in selected:
        for v in range(variant):
            suffix = f"_v{v}" if variant > 1 else ""
            out_path = os.path.join(output_dir, f"frame_{num:04d}{suffix}.png")
            if not os.path.exists(out_path):
                frames.append((num, v))

    return frames


def save_label(frame_num, variant_idx, category, hazard_description, variants_total):
    """Append entry to labels.json."""
    os.makedirs(os.path.dirname(LABELS_FILE), exist_ok=True)

    labels = {"images": []}
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r") as f:
            labels = json.load(f)

    suffix = f"_v{variant_idx}" if variants_total > 1 else ""
    entry = {
        "path": f"{category}/frame_{frame_num:04d}{suffix}.png",
        "category": category,
        "description": hazard_description,
        "source_frame": frame_num,
        "variant": variant_idx,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    labels["images"].append(entry)

    with open(LABELS_FILE, "w") as f:
        json.dump(labels, f, indent=2)


def process_frame(page, frame_num, variant_idx, category, variants_total, retry=0):
    """Upload realistic image to Gemini, inject hazard, download result."""
    input_path = os.path.join(INPUT_DIR, f"frame_{frame_num:04d}.png")
    output_dir = os.path.join(OUTPUT_BASE, category)
    os.makedirs(output_dir, exist_ok=True)

    suffix = f"_v{variant_idx}" if variants_total > 1 else ""
    save_path = os.path.join(output_dir, f"frame_{frame_num:04d}{suffix}.png")

    full_prompt, hazard_desc = assemble_prompt(category, frame_num)

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

        # ==== STEP 3: Upload image ====
        log(f"  [3/7] Uploading frame_{frame_num:04d}.png...")
        plus_btn = page.locator('button[aria-label="Open upload file menu"]').first
        plus_btn.wait_for(state="visible", timeout=5000)
        plus_btn.click()
        time.sleep(1)

        with page.expect_file_chooser(timeout=5000) as fc_info:
            page.get_by_text("Upload files", exact=True).first.click()
        fc_info.value.set_files(input_path)

        log(f"    Waiting for image to attach...")
        time.sleep(5)

        for check in range(10):
            previews = page.locator('img[src^="blob:"], .upload-preview, .file-preview').all()
            if previews:
                log(f"    Image attached!")
                break
            time.sleep(1)
        else:
            log(f"    WARNING: Could not confirm attachment, continuing...")
        time.sleep(1)

        # ==== STEP 4: Paste prompt ====
        log(f"  [4/7] Pasting prompt ({category})...")
        log(f"    Hazard: {hazard_desc[:80]}...")
        input_field = page.locator('div[contenteditable="true"]').first
        input_field.wait_for(state="visible", timeout=5000)
        input_field.click()
        time.sleep(0.5)

        page.evaluate("(text) => navigator.clipboard.writeText(text)", full_prompt)
        time.sleep(0.3)
        page.keyboard.press("Meta+v")
        time.sleep(1.5)

        content = input_field.inner_text()
        if len(content) < 50:
            log(f"    Clipboard paste failed, typing directly...")
            input_field.click()
            page.keyboard.press("Meta+a")
            page.keyboard.press("Backspace")
            page.keyboard.type(full_prompt, delay=1)
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

            ai_imgs = page.locator('img[alt*="AI generated"]').all()
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

            stopped = page.get_by_text("Response stopped").all()
            if stopped:
                log(f"    Gemini stopped response at {elapsed}s!")
                break

            if elapsed % 15 == 0:
                log(f"    Waiting... ({elapsed}s)")

        if not image_ready:
            log(f"    TIMEOUT after 180s")
            return False

        # ==== STEP 7: Download ====
        log(f"  [7/7] Downloading...")
        time.sleep(2)

        # Method 1: Overlay download button
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
            log(f"    SAVED (overlay): frame_{frame_num:04d}{suffix}.png ({file_size // 1024}KB)")
            save_label(frame_num, variant_idx, category, hazard_desc, variants_total)
            return True
        except Exception as e:
            log(f"    Overlay download failed: {e}")

        # Method 2: "..." menu → "Download image"
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
            log(f"    SAVED (menu): frame_{frame_num:04d}{suffix}.png ({file_size // 1024}KB)")
            save_label(frame_num, variant_idx, category, hazard_desc, variants_total)
            return True
        except Exception as e:
            log(f"    Menu download failed: {e}")

        return False

    except Exception as e:
        log(f"  ERROR: {e}")
        if retry < 2:
            log(f"  Retrying (attempt {retry + 2})...")
            time.sleep(5)
            return process_frame(page, frame_num, variant_idx, category, variants_total, retry + 1)
        return False


def run_injection(category, start, end, step, variants, delay, overnight):
    frames = get_frames_to_process(start, end, step, variants, category)
    if not frames:
        print("All frames already processed!")
        return

    with open(LOG_FILE, "a") as f:
        f.write(f"\n=== {category.upper()} Started {datetime.datetime.now()} ===\n")
        f.write(f"Frames: {len(frames)} items\n\n")

    log(f"{'='*60}")
    log(f"  HAZARD INJECTION: {category.upper()}")
    log(f"  {len(frames)} images to generate")
    log(f"  Input: {INPUT_DIR}")
    log(f"  Output: {os.path.join(OUTPUT_BASE, category)}")
    log(f"  Mode: {'OVERNIGHT' if overnight else 'INTERACTIVE'}")
    log(f"{'='*60}")

    with sync_playwright() as p:
        log("Connecting to Chrome on port 9222...")
        try:
            browser = p.chromium.connect_over_cdp("http://localhost:9222")
        except Exception as e:
            print(f"\nERROR: Could not connect to Chrome: {e}")
            print("\nMake sure Chrome is running with:")
            print('  /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome --remote-debugging-port=9222')
            return

        context = browser.contexts[0]
        page = context.new_page()
        page.goto("https://gemini.google.com/app")
        time.sleep(3)

        log("Connected! Checking login...")

        sign_in = page.get_by_text("Sign in", exact=True).all()
        if sign_in:
            print("\n  >>> Sign into Gemini in Chrome first, then press ENTER...")
            input()

        print(f"\n  >>> Ready to inject {category.upper()} hazards!")
        print(f"  >>> {len(frames)} images to process")
        print(f"  >>> Press ENTER to start...\n")
        input()

        success = 0
        failed = 0
        failed_frames = []
        consecutive_fails = 0
        RATE_LIMIT_THRESHOLD = 3  # 3 failures in a row = likely rate limited
        RATE_LIMIT_WAIT = 600     # Wait 10 minutes before retrying

        for i, (frame_num, variant_idx) in enumerate(frames):
            v_label = f" (variant {variant_idx + 1}/{variants})" if variants > 1 else ""
            log(f"\n[{i+1}/{len(frames)}] Frame {frame_num:04d}{v_label} → {category}")

            if process_frame(page, frame_num, variant_idx, category, variants):
                success += 1
                consecutive_fails = 0
            else:
                failed += 1
                consecutive_fails += 1
                failed_frames.append((frame_num, variant_idx))

                if consecutive_fails >= RATE_LIMIT_THRESHOLD:
                    log(f"\n  >>> {consecutive_fails} consecutive failures — likely RATE LIMITED")
                    log(f"  >>> Pausing for {RATE_LIMIT_WAIT // 60} minutes before retrying...")
                    log(f"  >>> Will resume at {(datetime.datetime.now() + datetime.timedelta(seconds=RATE_LIMIT_WAIT)).strftime('%H:%M:%S')}")
                    time.sleep(RATE_LIMIT_WAIT)
                    consecutive_fails = 0

                    # Retry the last failed frame after waiting
                    log(f"  >>> Retrying frame {frame_num:04d} after cooldown...")
                    if process_frame(page, frame_num, variant_idx, category, variants):
                        success += 1
                        failed -= 1
                        failed_frames.pop()
                        log(f"  >>> Retry succeeded! Continuing...")
                    else:
                        consecutive_fails = 1
                        log(f"  >>> Still failing after wait. Will try next frame...")
                elif not overnight:
                    log("  >> FAILED. Press ENTER to continue...")
                    input()
                else:
                    log("  >> FAILED (overnight mode, continuing...)")

            if (i + 1) % 10 == 0:
                log(f"\n  --- PROGRESS: {success} ok, {failed} failed, "
                    f"{len(frames) - i - 1} remaining ---")

            if i < len(frames) - 1:
                time.sleep(delay)

        log(f"\n{'='*60}")
        log(f"  DONE ({category})! Success: {success} | Failed: {failed}")
        if failed_frames:
            log(f"  Failed: {failed_frames}")
        log(f"{'='*60}")

        page.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject hazards into warehouse images via Gemini")
    parser.add_argument("--category", required=True,
                        choices=["spill", "missing_ppe", "forklift_violation", "improper_stacking", "obstacle"],
                        help="Hazard category to inject")
    parser.add_argument("--start", type=int, default=0, help="Start frame number")
    parser.add_argument("--end", type=int, default=170, help="End frame number")
    parser.add_argument("--step", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--variants", type=int, default=1, help="Variants per frame (for forklift)")
    parser.add_argument("--delay", type=int, default=15, help="Delay between frames (seconds)")
    parser.add_argument("--overnight", action="store_true", help="Skip failures without pausing")
    args = parser.parse_args()

    run_injection(args.category, args.start, args.end, args.step, args.variants, args.delay, args.overnight)
