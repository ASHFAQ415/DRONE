#!/usr/bin/env bash
set -euo pipefail

# Install the Debian/Trixie-friendly GPIO stack for Raspberry Pi servo PWM.
# This does not install or start pigpiod.

if ! command -v apt-get >/dev/null 2>&1; then
  echo "apt-get was not found. Run this script on Raspberry Pi OS."
  exit 1
fi

sudo apt-get update
sudo apt-get install -y python3-gpiozero python3-lgpio
python -m pip install -r requirements.txt

echo "gpiozero/lgpio setup complete. No pigpiod service is required."
echo "Use: export DRONE_SERVO_ENABLED=1"
