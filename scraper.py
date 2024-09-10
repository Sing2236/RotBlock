from googleapiclient.discovery import build

import json

# Replace with your API key
api_key = "AIzaSyB6Wfh1a--FT3v_V1i7pigHDU3PiqLeLNo"

youtube = build("youtube", "v3", developerKey=api_key)

def get_video_metadata(query, max_results=10):
    request = youtube.search().list(
        part="snippet",
        q=query,
        type="video",
        maxResults=max_results
    )
    response = request.execute()

    video_metadata = []
    for item in response['items']:
        video_metadata.append({
            'title': item['snippet']['title'],
            'description': item['snippet']['description'],
            'tags': item.get('snippet', {}).get('tags', []),
        })
    
    return video_metadata

# Example usage
metadata = get_video_metadata("brainrot", 20)

# Save the metadata to a JSON file
with open('video_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)

print("Data has been saved to video_metadata.json")
