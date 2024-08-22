#!/bin/bash

INTERFACE="eth0" 
OUTPUT_FILE="roundh3_20.txt" 

> $OUTPUT_FILE

capture_bandwidth() {
  bmon -p $INTERFACE -o ascii:noheader -c 1 --bits | awk '/^$INTERFACE/ {print $3}' >> $OUTPUT_FILE
}

while true; do
  for i in {1..4}; do
    capture_bandwidth
    sleep 0.25
  done
done
