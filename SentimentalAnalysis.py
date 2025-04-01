import os
from transformers import pipeline
import numpy as np

classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


def analyze_influencer_comments(target_directory, output_directory, max_folders=5, max_length=512):
    folder_count = 0
    for influencer_name in os.listdir(target_directory):
        influencer_folder = os.path.join(target_directory, influencer_name)
        if os.path.isdir(influencer_folder):
            folder_count += 1
            if folder_count > max_folders:
                break
            comments_file = os.path.join(influencer_folder, "comments.txt")
            if os.path.exists(comments_file):
                try:
                    with open(comments_file, "r", encoding="utf-8") as cmf:
                        comments = cmf.readlines()

                    # Concatenate all comments into a single string
                    combined_comments = " ".join([comment.strip() for comment in comments if comment.strip()])

                    # If combined comments exceed max_length, split them into smaller chunks
                    sentiment_results = []
                    for i in range(0, len(combined_comments), max_length):
                        comment_chunk = combined_comments[i:i + max_length]
                        sentiment = classifier(comment_chunk)
                        sentiment_results.append(sentiment[0])

                    # Aggregate sentiment results by taking the majority sentiment
                    sentiment_labels = [result['label'] for result in sentiment_results]
                    sentiment_scores = [result['score'] for result in sentiment_results]

                    # Majority sentiment
                    majority_sentiment = max(set(sentiment_labels), key=sentiment_labels.count)

                    # Average sentiment score
                    average_score = np.mean(sentiment_scores)

                    output_file = os.path.join(output_directory, f"{influencer_name}_sentiment_analysis.txt")
                    os.makedirs(output_directory, exist_ok=True)

                    with open(output_file, "w", encoding="utf-8") as classf:
                        classf.write(f"Majority sentiment for combined comments: {majority_sentiment}\n")
                        classf.write(f"Average sentiment score: {average_score}\n")
                except Exception as e:
                    print(f"Error processing {influencer_name}: {e}")
            else:
                print(f"Comments file for {influencer_name} not found.")


analyze_influencer_comments(r"C:\Users\Adithya\PycharmProjects\Capstone\Processed", r"C:\Users\Adithya\PycharmProjects\Capstone\Sentiment_Analysis")
