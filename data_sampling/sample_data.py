#!/usr/bin/env python3

import os

# File names
influencers_in = "influencers.txt"
mappings_in = "JSON-Image_files_mapping.txt"
influencers_out = "smallInfluencers.txt"
mappings_out = "smallMappings.txt"
INFLUENCERS_PER_CATEGORY = 200
POSTS_PER_INFLUENCER = 5

# First pass: Process influencers.txt to select at most INFLUENCERS_PER_CATEGORY influencers per category
# and, for each selected influencer, only the first POSTS_PER_INFLUENCER posts.
# We assume columns are tab-separated and that the influencer name is the first column
# and the category is the second column.
selected_influencers = {}  # { category: set([influencer, ...]) }
post_counts = {}  # { influencer: count } to count posts written

with open(influencers_in, "r", encoding="utf-8") as inf_in, open(
    influencers_out, "w", encoding="utf-8"
) as inf_out:

    for line in inf_in:
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        influencer = parts[0]
        category = parts[1]

        # If this category is not seen, initialize its set
        if category not in selected_influencers:
            selected_influencers[category] = {}

        # If influencer is not selected yet, check if we can add it
        if influencer not in selected_influencers[category]:
            if len(selected_influencers[category]) < INFLUENCERS_PER_CATEGORY:
                selected_influencers[category][influencer] = 0
            else:
                # Skip posts from influencers that are not in the selected set
                continue

        # Only write if the influencer has not reached POSTS_PER_INFLUENCER posts yet.
        if selected_influencers[category][influencer] < POSTS_PER_INFLUENCER:
            inf_out.write(line + "\n")
            selected_influencers[category][influencer] += 1

# For convenience, create a lookup set of all selected influencers (with counts reset)
selected_lookup = set()
for cat, infl_dict in selected_influencers.items():
    for infl in infl_dict.keys():
        selected_lookup.add(infl)

# Second pass: Process the mapping file.
# We assume that for each influencer, the mapping file lines are in the same order as the influencers file.
# We only include mapping lines for influencers in our selected set, and only the first POSTS_PER_INFLUENCER occurrences.
mapping_counts = {}  # { influencer: count }

with open(mappings_in, "r", encoding="utf-8") as map_in, open(
    mappings_out, "w", encoding="utf-8"
) as map_out:

    for line in map_in:
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 1:
            continue
        influencer = parts[0]
        if influencer not in selected_lookup:
            continue

        # Get current count for this influencer, defaulting to 0
        count = mapping_counts.get(influencer, 0)
        if count < POSTS_PER_INFLUENCER:
            map_out.write(line + "\n")
            mapping_counts[influencer] = count + 1

print("Finished writing", influencers_out, "and", mappings_out)
