from googleapiclient.discovery import build
import json
import csv
import pytube
from pytube import YouTube
# Replace with your API key
api_key = "AIzaSyB6Wfh1a--FT3v_V1i7pigHDU3PiqLeLNo"

youtube = build("youtube", "v3", developerKey=api_key)

# List of keywords or tags related to "brain rot"
brain_rot_tags = [
    "shorts", "trending", "funny", "viral", "satisfying", "challenge", 
    "fyp", "for you", "memes", "life hacks"
]

# Function to check if video matches "brain rot" characteristics
def is_brain_rot(video, tags):
    snippet = video.get('snippet')
    if not snippet:
        return False  # Skip this video if 'snippet' is missing
    
    title = snippet.get('title', '').lower()
    description = snippet.get('description', '').lower()
    video_tags = snippet.get('tags', [])
    view_count = video.get('statistics', {}).get('viewCount', 0)
    duration = video.get('contentDetails', {}).get('duration', '')

    # Check for tags or phrases in title, description, or tags
    content = f"{title} {description} {' '.join(video_tags)}"
    has_brain_rot_tag = any(tag in content for tag in tags)

    # Short video length (PT format is ISO 8601 duration)
    short_video = 'PT' in duration and ('S' in duration or 'M' in duration and '1M' in duration)
    
    # Video with a high view count for short content
    viral_video = int(view_count) > 100000 if view_count else False
    
    return has_brain_rot_tag and short_video and viral_video

# Get video metadata with content details and statistics
def get_video_metadata(query, max_results=10):
    request = youtube.search().list(
        part="snippet",
        q=query,
        type="video",
        maxResults=max_results
    )
    response = request.execute()

    video_ids = [item['id']['videoId'] for item in response['items']]

    # Get more details on each video (duration, views)
    details_request = youtube.videos().list(
        part="contentDetails,statistics,snippet",
        id=','.join(video_ids)
    )
    details_response = details_request.execute()

    video_metadata = []
    for video in details_response['items']:
        brain_rot_detected = is_brain_rot(video, brain_rot_tags)
        video_metadata.append({
            'title': video.get('snippet', {}).get('title', 'N/A'),
            'description': video.get('snippet', {}).get('description', 'N/A'),
            'tags': video.get('snippet', {}).get('tags', []),
            'url': f"https://www.youtube.com/watch?v={video['id']}",
            'viewCount': video.get('statistics', {}).get('viewCount', 0),
            'duration': video.get('contentDetails', {}).get('duration', 'N/A'),
            'brain_rot_detected': brain_rot_detected
        })

    return video_metadata

# Collect metadata
metadata = get_video_metadata("funny shorts", 20)

# Save the metadata to a JSON file
with open('video_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)

# Save to CSV
def save_metadata_to_csv(metadata, filename):
    keys = metadata[0].keys()  # Extract headers
    with open(filename, 'w', newline='', encoding='utf-8') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(metadata)

save_metadata_to_csv(metadata, 'youtube_videos.csv')

print("Data has been saved to video_metadata.json and youtube_videos.csv")
