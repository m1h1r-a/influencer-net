import os
from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
categories = ['beauty', 'family', 'fashion', 'fitness', 'food', 'interior', 'pet', 'travel', 'other']

def classify_influencer_captions(target_directory, output_directory, max_folders=5):
    folder_count = 0
    for influencer_name in os.listdir(target_directory):
        influencer_folder = os.path.join(target_directory, influencer_name)
        if os.path.isdir(influencer_folder):
            folder_count += 1
            if folder_count > max_folders:
                break
            caption_file = os.path.join(influencer_folder, "captions.txt")
            if os.path.exists(caption_file):
                try:
                    with open(caption_file, "r", encoding="utf-8") as cf:
                        captions = cf.readlines()
                    combined_captions = " ".join([caption.strip() for caption in captions if caption.strip()])
                    if combined_captions:
                        result = classifier(combined_captions, categories)
                        classified_category = result["labels"][result["scores"].index(max(result["scores"]))]

                        output_file = os.path.join(output_directory, f"{influencer_name}_classification.txt")
                        os.makedirs(output_directory, exist_ok=True)

                        with open(output_file, "w", encoding="utf-8") as classf:
                            classf.write(f"Combined captions classified as: {classified_category}\n")
                            classf.write(f"Full classification result: {result}")
                except Exception as e:
                    print(f"Error processing {influencer_name}: {e}")
            else:
                print(f"Caption file for {influencer_name} not found.")

classify_influencer_captions(r"C:\Users\Adithya\PycharmProjects\Capstone\Processed", r"C:\Users\Adithya\PycharmProjects\Capstone\Classified")
