#!/usr/bin/env python3

import os

# File names
influencers_in = "influencers.txt"
mappings_in = "JSON-Image_files_mapping.txt"
influencers_out = "smallInfluencers.txt"
mappings_out = "smallMappings.txt"
INFLUENCERS_PER_CATEGORY = 50
POSTS_PER_INFLUENCER = 300

# select x influncers per category and first y posts per influeners
selected_influencers = {}  # { category: set([influencer, ...]) }
post_counts = {}  # { influencer: count }

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

        # initialize if new category
        if category not in selected_influencers:
            selected_influencers[category] = {}

        # add influencer if not in set
        if influencer not in selected_influencers[category]:
            if len(selected_influencers[category]) < INFLUENCERS_PER_CATEGORY:
                selected_influencers[category][influencer] = 0
            else:
                continue

        # write till y posts per influencer
        if selected_influencers[category][influencer] < POSTS_PER_INFLUENCER:
            inf_out.write(line + "\n")
            selected_influencers[category][influencer] += 1

# dict with all data
selected_lookup = set()
for cat, infl_dict in selected_influencers.items():
    for infl in infl_dict.keys():
        selected_lookup.add(infl)

# process the mapping file
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

        # get current count for this influencer, defaulting to 0
        count = mapping_counts.get(influencer, 0)
        if count < POSTS_PER_INFLUENCER:
            map_out.write(line + "\n")
            mapping_counts[influencer] = count + 1

print("Finished writing", influencers_out, "and", mappings_out)
