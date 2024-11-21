//fires when navigating youtube
document.addEventListener('yt-navigate-finish', () => {
    chrome.storage.local.get('rotblockActive', function(data) {
        if (data.rotblockActive === 'true' && document.readyState !== 'loading'){
            console.log('doc ready, starting scraper')
            startScraping();
        }else{
            document.addEventListener('DOMContentLoaded', function () {
                console.log('document was not ready, starting scraper once it is');
                startScraping();
            });
        }
    });
});

//fires when receiving a message from popup.js (usually because scraper is off)
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "startScraping") {
        startScraping();
    }
});

// Function to start the scraping process
function startScraping() {
    console.log('Scraper activated');
    extractVideoData();
}

// Function to extract video data from YouTube, currently the title but creator and url are possible
function extractVideoData() {
    const videoData = [];
    //grab html element that contains title and then get the title from the element
    const titleElement = document.querySelector('h1.style-scope.ytd-watch-metadata');
    videoTitle = titleElement.textContent.trim();
    //get creator
    videoCreator = getVideoCreator();
    //get url
    videoUrl = document.URL

    //then push 'em all onto the video data stack
    if (videoUrl && videoTitle && videoCreator) {
        videoData.push({
            url: videoUrl,
            title: videoTitle,
            creator: videoCreator
        });
    }

    // If video data is found, download as CSV
    //if (videoData.length > 0) {
    //    downloadCSV(videoData);
    //} else {
    //    console.log("No video data found.");
    //}

    //If no video data somehow ends up on the stack, something has gone fuckin wrong and we must abort
    if (videoData.length === 0){
        console.log("No video data found. Something's gone wrong.");
        return;
    }

    sendVideoData(videoTitle);

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

function sendVideoData(data) {
    fetch('http://127.0.0.1:5000/getBrainrot', {
        method: 'POST',
        headers: {
            'Content-Type': 'text/plain'
        },
        body: data
    });

    //THIS DOESN'T WORK!
    //look into biting the bullet and just using jQuery so you can send ajax requests
}

