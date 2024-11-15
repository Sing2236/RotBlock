// Function to start the scraping process
function startScraping() {
    console.log('Scraper activated');
    extractVideoData();
}

// Function to extract video data from YouTube
function extractVideoData() {
    const videoData = [];

    // Find all <a> tags that contain "watch?v" in their href
    const videoLinks = document.querySelectorAll('a[href*="watch?v="]');

    // Loop through each link and extract the href and title
    videoLinks.forEach(link => {
        const videoUrl = link.href;
        let videoTitle = link.getAttribute('title') || link.textContent.trim();

        // Clean up the title by removing anything related to video length or other irrelevant text
        videoTitle = videoTitle.replace(/(\d{1,2}[:]\d{2})|Now playing/g, '').trim();

        // Extract creator from the page (if on a video page)
        const creator = getVideoCreator();

        // If we have a valid video URL and title, push to the videoData array
        if (videoUrl && videoTitle && creator) {
            videoData.push({
                url: videoUrl,
                title: videoTitle,
                creator: creator
            });
        }
    });

    // If video data is found, download as CSV
    if (videoData.length > 0) {
        downloadCSV(videoData);
    } else {
        console.log("No video data found.");
    }
}

// Function to extract the creator's name from the page
function getVideoCreator() {
    const creator = document.querySelector("#upload-info a")?.innerText || '';
    return creator;
}

// Function to download the data as CSV
function downloadCSV(data) {
    const csvContent = 'Video URL, Video Title, Creator\n' + data.map(item => {
        return `"${item.url}","${item.title}","${item.creator.replace(/"/g, '""')}"`;
    }).join('\n');

    // Create a Blob with CSV data
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });

    // Create a download link and trigger the download
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'youtube_video_metadata.csv';

    // Append the link to the DOM and trigger a click to download the CSV
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "startScraping") {
        startScraping();
    }
});
