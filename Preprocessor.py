import os
import json


def process_info_files(source_directory, target_directory):
    for filename in os.listdir(source_directory):
        if filename.endswith(".info"):
            filepath = os.path.join(source_directory, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                data = json.load(file)

            influencer_name = filename.split("-")[0]
            influencer_folder = os.path.join(target_directory, influencer_name)
            os.makedirs(influencer_folder, exist_ok=True)

            caption_file = os.path.join(influencer_folder, "captions.txt")
            with open(caption_file, "a", encoding="utf-8") as cf:
                captions = [edge["node"]["text"] for edge in data.get("edge_media_to_caption", {}).get("edges", [])]
                if captions:
                    cf.write("\n".join(captions) + "\n")

            comments_file = os.path.join(influencer_folder, "comments.txt")
            with open(comments_file, "a", encoding="utf-8") as cmf:
                comments = [edge["node"]["text"] for edge in
                            data.get("edge_media_to_parent_comment", {}).get("edges", [])]
                if comments:
                    cmf.write("\n".join(comments) + "\n")


process_info_files(r"C:\Users\Adithya\PycharmProjects\Capstone\data_sampling\sampled_data\info", r"C:\Users\Adithya\PycharmProjects\Capstone\Processed")