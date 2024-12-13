#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

echo "Updating the server..."
sudo apt-get update

echo "Checking Python version..."

# Required Python version
REQUIRED_PYTHON_VERSION="3.12.8"

# Check if the required Python version is installed
if ! python3 --version | grep -q "$REQUIRED_PYTHON_VERSION"; then
    echo "Python $REQUIRED_PYTHON_VERSION is not installed. Installing required Python version..."
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install -y python${REQUIRED_PYTHON_VERSION%.*} python${REQUIRED_PYTHON_VERSION%.*}-venv python${REQUIRED_PYTHON_VERSION%.*}-dev
fi

echo "Ensuring Python $REQUIRED_PYTHON_VERSION is being used..."
PYTHON_EXECUTABLE="/usr/bin/python${REQUIRED_PYTHON_VERSION%.*}"

# Verify the correct Python version is being used
if ! $PYTHON_EXECUTABLE --version | grep -q "$REQUIRED_PYTHON_VERSION"; then
    echo "Failed to set Python $REQUIRED_PYTHON_VERSION. Please check your system setup."
    exit 1
fi

echo "Installing pip..."
sudo apt install -y python3-pip

echo "Installing virtual environment and packages..."
cd LiveTradingBots/code

# Create virtual environment using the required Python version
$PYTHON_EXECUTABLE -m venv .venv
source .venv/bin/activate

# Upgrade pip and install requirements
pip install --upgrade pip
pip install -r ../requirements.txt

cd ..

echo "Setup complete! Activate your virtual environment with:"
echo "source LiveTradingBots/code/.venv/bin/activate"
