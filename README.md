# A simple, low-budget and scalable method for automatically tracking movement in arthropods

This repository contains the code and 3D-print files for the system described in:

> Neefjes et al., *A simple, low-budget and scalable method for automatically tracking movement in arthropods* (in preparation)

The project provides a Raspberry Pi–based imaging setup that automatically quantifies arthropod movement over time and can be used to derive activity patterns and survival curves under laboratory conditions.

---

## Overview

The system is designed to:

- Monitor small arthropods continuously over several days using visible and near‑infrared light.
- Quantify movement objectively from images as **pixel-change–based activity scores**.
- Operate at low hardware cost and be easily scaled to many individuals and enclosures.

Core components:

- Raspberry Pi 5 (2 GB RAM) as single-board computer  
- Two Raspberry Pi Camera Module 3 NoIR (12 MP)  
- Near‑infrared LED arrays (≈ 880 nm) for night-time imaging  
- Tube lights controlled by a climate chamber for daytime imaging  
- Custom Python code for image acquisition and activity quantification  
- 3D-printed mounts and tube racks for reproducible positioning of cameras and enclosures  

The system was validated on workers of *Formica polyctena* and larvae of *Alphitobius diaperinus* in glass tubes and petri dishes, under different treatments.

---

## Repository contents

- `imaging_code.py`  
  Main Python script that:
  - Configures the Raspberry Pi cameras (focus, ISO, exposure, white balance).
  - Asks the user for setup name, camera names and day–night cycle.
  - Lets the user define Regions of Interest (ROIs) for each individual.
  - Captures images at fixed intervals and compares them pixel-by-pixel.
  - Applies a Gaussian blur, converts images to grayscale, and computes pixel change per ROI.
  - Normalizes pixel change per individual to an **activity score** between 0 and 1.
  - Saves activity data to CSV files on the SD card and uploads them to cloud storage.

- `Small_Ant_Tube_RackV2 (3).stl`  
  3D model of the horizontal tube rack made from PMMA in the study. Designed to hold 5 ml glass test tubes and provide a standardized, high-contrast background.

- `Pi_Module3_Case (1).stl`  
  3D model of a case/mount for the Raspberry Pi Camera Module 3, allowing stable top‑down positioning above the experimental setup.

- `README.md`  
  This file.

> Note: Filenames may contain spaces or parentheses depending on export settings; rename them if needed before printing.

---

## Hardware requirements (example setup)

The system has been tested with:

- Raspberry Pi 5 (2 GB RAM)  
- SanDisk Ultra 64 GB microSD card  
- 2 × Raspberry Pi Camera Module 3 NoIR  
- 2 × 12 W near‑infrared LED arrays (dominant wavelength ≈ 880 nm)  
- 2 × 12 V / 1 A power supplies for the IR LEDs  
- Visible-light tube lamps (e.g. F58W/2084-t8) controlled by a climate chamber  
- Wooden or other stable frame to mount cameras at ~60 cm above the enclosures  
- 3D‑printed tube racks and camera mounts (`*.stl` from this repository)

You can adapt the hardware to your own laboratory setup as long as:

- Cameras have sufficient resolution and near‑IR sensitivity.
- Illumination is stable and produces adequate contrast between organism and background.
- The Raspberry Pi has enough RAM (≥ 1.2 GB) to run the imaging program.

---

## Software requirements

- Raspberry Pi OS (or another Linux distribution compatible with Raspberry Pi 5)
- Python 3 (version depends on your Pi OS; see `imaging_code.py` for details)
- Required Python libraries: see the import statements in `imaging_code.py`

Because the exact dependency versions may change during development, please consult `imaging_code.py` directly and install the listed packages using `pip` or your package manager.

---

## Installation and setup (high level)

1. **Clone or download this repository** on your workstation or directly on the Raspberry Pi:
   ```bash
   git clone https://github.com/robertneefjes/A-simple-low-budget-and-scalable-method-for-automatically-tracking-movement-in-arthropods.git
