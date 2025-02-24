#!/bin/bash

# Set initial exposure to 0
cam apx60 exposure 0
echo "Exposure set to 0."
sleep 2  # Wait for 2 seconds
pipeline reset
sleep 2
pipeline prefix bias # Set prefix to bias
echo "Prefix set to bias."

echo "Starting script."

# Start the loop, running for 1 minute
end_time=$((SECONDS+60))

while [ $SECONDS -lt $end_time ]; do
    # Enable image saving
    pipeline archive apx60 enable
    sleep 2
    # Take one image
    cam apx60 start 1
    sleep 2
    # Disable image saving
    pipeline archive apx60 disable
    sleep 2
    # Take another image
    cam apx60 start 1
    sleep 2
    # Enable image saving again
    pipeline archive apx60 enable
    sleep 2
    # Take one more image
    cam apx60 start 1
done

# Print message after the script finishes
echo "Finished script."