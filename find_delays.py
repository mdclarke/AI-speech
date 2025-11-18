#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 22:58:29 2025

@author: maggie

Get delays between MEG triggers and MISC audio output
"""

import mne
import numpy as np

fname = '/home/maggie/data/delaytest_raw.fif'
raw = mne.io.read_raw_fif(fname, preload=True)
sfreq = raw.info['sfreq']

# Detect MISC002 rising onsets
misc_data = raw.copy().pick_channels(['MISC002']).get_data()[0]

# Detection parameters
threshold = 0.06 # amplitude (rising edge)
pre_threshold = 0.005  # baseline close to zero 
refractory_ms = 300    # minimum spacing between events
refractory_samples = int((refractory_ms / 1000) * sfreq)

# Find all rising threshold crossings
onset_samples = []
i = 1
while i < len(misc_data):
    if misc_data[i-1] <= pre_threshold and misc_data[i] >= threshold:
        onset_samples.append(i)
        i += refractory_samples  # skip ahead to avoid duplicates
    else:
        i += 1

# Convert to times in seconds
onset_times_sec = np.array(onset_samples) / sfreq

print(f"Found {len(onset_times_sec)} rising events on MISC002")
print("First 5 MISC002 onsets (s):", onset_times_sec[:5])

## Detect STI101 triggers
stim_channel = 'STI101'
first_samp = raw.first_samp  # sample index offset
print(f"raw.first_samp: {first_samp}")

events = mne.find_events(raw, stim_channel=stim_channel, output='onset',
                         shortest_event=1)
trigger_samples = events[:, 0]
adjusted_samples = trigger_samples - first_samp  # align with raw data indexing
trigger_times_sec = adjusted_samples / sfreq

print(f"Found {len(trigger_times_sec)} triggers on {stim_channel}")
print("First 5 STI101 triggers (s):", trigger_times_sec[:5])

delay = trigger_times_sec - onset_times_sec
average_delay_ms = np.mean(delay) * 1000
median_delay_ms = np.median(delay) * 1000
jitter_ms = np.std(delay) * 1000

print(f"Average delay: {average_delay_ms:.2f} ms")
print(f"Median delay: {median_delay_ms:.2f} ms")
print(f"Jitter (std dev): {jitter_ms:.2f} ms")