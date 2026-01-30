# Warehouse Hazard Scenarios

Reference document for scene creation. Each scenario = training data for detection model.

---

## 1. SPILLS (Class: `spill`)

Liquid or substance on floor creating slip hazard.

| ID | Scenario | Description |
|----|----------|-------------|
| S1 | Water puddle | Clear water spill, reflective |
| S2 | Oil spill | Dark, slick oil patch |
| S3 | Chemical spill | Colored liquid (green, yellow) |
| S4 | Leaked product | Spill from damaged package |
| S5 | Large spill | Covers significant floor area |
| S6 | Small spill | Small puddle, easy to miss |
| S7 | Spill near equipment | Puddle near forklift/pallet jack |
| S8 | Spill in aisle | Blocking walking path |

---

## 2. OBSTACLES IN AISLES (Class: `obstacle`)

Objects blocking pathways or creating trip hazards.

| ID | Scenario | Description |
|----|----------|-------------|
| O1 | Box in aisle | Single cardboard box blocking path |
| O2 | Multiple boxes | Several boxes scattered in aisle |
| O3 | Fallen pallet | Pallet lying flat in walkway |
| O4 | Debris on floor | Wood planks, packaging materials |
| O5 | Equipment left out | Pallet jack left in middle of aisle |
| O6 | Shrink wrap on floor | Plastic wrap creating trip hazard |
| O7 | Leaning objects | Items propped against shelving |
| O8 | Partially blocked aisle | Narrow passage due to objects |

---

## 3. MISSING PPE (Class: `missing_ppe`)

Workers without required safety equipment.

| ID | Scenario | Description |
|----|----------|-------------|
| P1 | No hard hat | Worker without hard hat in hard hat zone |
| P2 | No safety vest | Worker without high-vis vest |
| P3 | No hard hat + no vest | Worker missing both |
| P4 | Hard hat off | Hard hat on head but not worn properly (in hand, on shelf) |
| P5 | Near forklift no PPE | Worker near forklift without proper PPE |
| P6 | Multiple workers no PPE | Group of workers, some missing PPE |

**Contrast scenes (SAFE - for model to learn difference):**
| ID | Scenario | Description |
|----|----------|-------------|
| P-SAFE1 | Full PPE | Worker with hard hat AND vest |
| P-SAFE2 | Team with PPE | Multiple workers all wearing PPE |

---

## 4. FORKLIFT VIOLATIONS (Class: `forklift_violation`)

Unsafe forklift operation or positioning.

| ID | Scenario | Description |
|----|----------|-------------|
| F1 | Forks raised empty | Forks elevated high with nothing on them |
| F2 | Forks raised while moving | Forklift driving with forks up |
| F3 | Person too close | Worker standing within danger zone of operating forklift |
| F4 | Pedestrian in path | Person in forklift's travel path |
| F5 | Overloaded forklift | Too many pallets/weight on forks |
| F6 | Unstable load | Load tilted, about to fall |
| F7 | Forklift in pedestrian area | Forklift where it shouldn't be |
| F8 | Speeding (motion blur) | Forklift moving too fast |
| F9 | No operator | Forklift left running/unattended with forks up |
| F10 | Blocked visibility | Forklift carrying load that blocks driver view |

---

## 5. BLOCKED EXITS (Class: `blocked_exit`)

Emergency exits obstructed or inaccessible.

| ID | Scenario | Description |
|----|----------|-------------|
| E1 | Boxes blocking exit | Cardboard boxes in front of exit door |
| E2 | Pallet blocking exit | Pallet placed in front of emergency exit |
| E3 | Equipment blocking | Forklift/pallet jack parked at exit |
| E4 | Partially blocked | Exit accessible but obstructed |
| E5 | Debris at exit | Trash, packaging blocking exit path |
| E6 | Shelving too close | Racking positioned blocking exit access |

---

## 6. IMPROPER STACKING (Class: `improper_stacking`)

Boxes, pallets, or items stacked unsafely creating falling object risk.

| ID | Scenario | Description |
|----|----------|-------------|
| IS1 | Stacked too high | Boxes stacked beyond safe height limit |
| IS2 | Leaning stack | Stack tilting, about to fall |
| IS3 | Unstable pyramid | Smaller boxes on larger, top-heavy |
| IS4 | Mixed sizes poorly stacked | Different box sizes stacked unsafely |
| IS5 | Heavy on top | Heavy items stacked on lighter items |
| IS6 | No pallet base | Boxes stacked directly on floor, unstable |
| IS7 | Overhanging stack | Stack extending past pallet edge |
| IS8 | Damaged boxes in stack | Crushed boxes supporting weight above |

---

## 7. FALL HAZARDS (Class: `fall_hazard`)

Workers at risk of falling from height.

| ID | Scenario | Description |
|----|----------|-------------|
| FH1 | Climbing racking | Worker using shelves as ladder |
| FH2 | Standing on pallet | Worker standing on pallet on forks |
| FH3 | Ladder wrong angle | Ladder too steep or too shallow |
| FH4 | Top step use | Worker standing on top step of ladder |
| FH5 | Overreaching on ladder | Worker leaning too far to side |
| FH6 | Ladder on uneven surface | Ladder on unstable ground |
| FH7 | No three-point contact | Worker not holding ladder properly |
| FH8 | Standing on equipment | Worker standing on forklift/pallet jack |

---

## 8. FIRE SAFETY HAZARDS (Class: `fire_hazard`)

Fire safety equipment blocked or improper storage of flammables.

| ID | Scenario | Description |
|----|----------|-------------|
| FS1 | Blocked fire extinguisher | Boxes/pallets blocking extinguisher access |
| FS2 | Blocked sprinkler | Items too close to ceiling sprinklers |
| FS3 | Flammables near ignition | Flammable materials near electrical/heat |
| FS4 | Improper chemical storage | Chemicals not in designated area |
| FS5 | Blocked fire hose | Fire hose cabinet obstructed |
| FS6 | Combustibles in aisle | Cardboard/packaging piled up |
| FS7 | No clear path to extinguisher | Route to fire equipment blocked |

---

## 9. MISSING SAFETY BARRIERS (Class: `missing_barrier`)

Safety gates, barriers, or guards not in place during operations.

| ID | Scenario | Description |
|----|----------|-------------|
| B1 | Aisle gate open | Safety gate at aisle end open while forklift operating |
| B2 | Mezzanine gate open | Upper level gate open during operations |
| B3 | Loading dock gate open | Dock barrier not in place |
| B4 | Rack guard missing | Floor-level rack protector missing/damaged |
| B5 | Forklift in area, gate open | Reach truck or forklift working with safety gate not closed |
| B6 | Pedestrian gate open | Gate separating pedestrian and forklift zones open |
| B7 | No barrier at drop-off | Edge of elevated area without barrier |

---

## 10. DAMAGED RACKING (Class: `damaged_rack`)

Structural damage to shelving/racking systems.

| ID | Scenario | Description |
|----|----------|-------------|
| R1 | Bent upright | Vertical post bent/dented |
| R2 | Bent beam | Horizontal beam bent |
| R3 | Missing safety pins | Beam clips/pins missing |
| R4 | Overloaded shelf | Shelf bowing under weight |
| R5 | Leaning rack | Entire rack unit tilted |
| R6 | Impact damage | Fresh damage marks (paint scraped, dents) |
| R7 | Rust/corrosion | Structural rust on supports |
| R8 | Unstable load on rack | Items about to fall off shelf |
| R9 | Item hanging off edge | Box/pallet not fully on shelf |
| R10 | Too much overhang | Pallet extending too far past shelf edge |

---

## 11. COMBINATION HAZARDS (Multiple classes)

Real warehouses often have multiple hazards in one scene.

| ID | Combination | Classes |
|----|-------------|---------|
| C1 | Spill + no wet floor sign + worker nearby | spill, obstacle |
| C2 | Forklift + worker too close + no PPE | forklift_violation, missing_ppe |
| C3 | Damaged rack + items falling + spill below | damaged_rack, spill, obstacle |
| C4 | Blocked exit + forklift parked there | blocked_exit, forklift_violation |
| C5 | Cluttered aisle + worker without vest | obstacle, missing_ppe |

---

## 12. SAFE SCENES (Negative examples)

Model needs to learn what "safe" looks like too.

| ID | Scenario | Description |
|----|----------|-------------|
| SAFE1 | Clean aisle | Clear walkway, nothing blocking |
| SAFE2 | Organized shelves | Proper storage, nothing overhanging |
| SAFE3 | Forklift proper | Forks down, stable load, no people nearby |
| SAFE4 | Worker with PPE | Full safety gear worn correctly |
| SAFE5 | Clear exit | Exit door fully accessible |
| SAFE6 | Clean floor | No spills, no debris |

---

## Scene Variations

For each hazard, vary these elements:

### Camera Angles
- Ground level (worker POV)
- Elevated (security camera POV)
- Side angle
- Down the aisle
- Close-up on hazard
- Wide shot showing context

### Lighting Conditions
- Bright (daytime, all lights on)
- Dim (some lights off, evening)
- Mixed (natural + artificial)
- Shadows (harsh overhead lighting)

### Warehouse Areas
- Main aisle
- Cross aisle
- Near loading dock
- Near exit
- Storage area
- High-traffic zone

---

## Priority for Initial Dataset

Start with these high-impact scenarios:

### Phase 1 (Must have)
- [ ] S1, S2 - Basic spills
- [ ] O1, O2, O4 - Boxes and debris in aisle
- [ ] P1, P2 - Missing hard hat, missing vest
- [ ] F1, F3 - Forks raised, person too close
- [ ] E1, E2 - Boxes/pallets blocking exit
- [ ] R1, R9 - Bent upright, item hanging off edge
- [ ] SAFE scenes for contrast

### Phase 2 (Add variety)
- [ ] All remaining scenarios
- [ ] Combination hazards
- [ ] Different lighting
- [ ] Multiple camera angles

---

## Notes

- Each scenario should be rendered from 3-5 camera angles
- Include "safe" versions for training contrast
- Stable Diffusion will add realism variations
- Target: 500+ base renders → 2000+ with SD enhancement
