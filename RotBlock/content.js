//gets video player element so it pauses before evaluation
function grabVideoPlayer() {
    if (document.URL.match(/.:\/\/www\.youtube\.com\/watch./i) || document.URL.match(/.:\/\/www\.youtube\.com\/shorts./i)){
        player = document.querySelector("video") ; 
        return player ;
    }
    else{
        console.log('couldnt grab, returning null') ;
        return null ;
    }
}

//called here for global variable
vidPlayer = grabVideoPlayer();

//fires when navigating youtube
document.addEventListener('yt-navigate-finish', () => {
    chrome.storage.local.get('rotblockActive', function(data) {
        if (data.rotblockActive === 'true' && document.readyState === 'complete'){
            console.log('doc ready, starting scraper - if nothing happened thats cause youre not on a yt vid')
            startScraping();
        }else{
            document.addEventListener('DOMContentLoaded', function () {
                console.log('document was not ready, starting scraper once it is - if nothing happened thats cause youre not on a yt vid ');
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

//function to sanitize input
function sanitize(string) {
    const map = {
        '&': 'amp',
        '<': 'lt',
        '>': 'gt',
        '"': 'quot',
        "'": 'squot',
        "/": 'slash',
    };
    const reg = /[&<>"'/]/ig;
    return string.replace(reg, (match)=>(map[match]));
  }

// Function to start the scraping process
function startScraping() {
    //uses regex to check if we're on a yt vid or short. then grabs the player
    if (document.URL.match(/.:\/\/www\.youtube\.com\/watch./i) || document.URL.match(/.:\/\/www\.youtube\.com\/shorts./i)){
        console.log('Scraper activated');
        vidPlayer = grabVideoPlayer();
        vidPlayer.addEventListener('loadeddata', function() {
            vidPlayer.pause();
            extractVideoData();
        }, false);
    }
    
}

// Function to extract video data from YouTube, currently the title but creator and url are possible
function extractVideoData() {
    //sometimes it doesn't pause when startScraping is called so it's here for redundancy
    videoTitle = null;
    vidPlayer.pause();
    const videoData = [];
    //console.log(retry)
    //grab html element that contains title and then get the title from the element
    const titleElement = document.querySelector('h1.style-scope.ytd-watch-metadata');
    videoTitle = titleElement.textContent.trim();
    //get creator
    videoCreator = getVideoCreator();
    //get url
    videoUrl = document.URL ;

    //then push 'em all onto the video data stack
    if (videoUrl && videoTitle && videoCreator) {
        videoData.push({
            url: videoUrl,
            title: videoTitle,
            creator: videoCreator
        });
    }
    else {
        console.log('Failed to grab all required metadata, retrying.');
        if (document.URL.match(/.:\/\/www\.youtube\.com\/watch./i) || document.URL.match(/.:\/\/www\.youtube\.com\/shorts./i)) {
            location.reload();
        }
    }


    // If video data is found, download as CSV (bit of legacy code for debugging)
    //if (videoData.length > 0) {
    //    downloadCSV(videoData);
    //} else {
    //    console.log("No video data found.");
    //}
    
    //this bit of code is redundant but should catch all cases where the else case
    //for above when nothing gets pushed to the stack just somehow doesn't
    //activate.

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
    //sends title data
    //console.log(data);
    input = sanitize(data); //have to sanitize input or else it'll cause errors w/ the ai's evaluation
    //console.log(input)

    //sends title to ai and gets back brainrot score
    //catch element is there to unpause the video incase ai is down
    fetch('http://127.0.0.1:5000/getBrainrot', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded'
        },
        body: 'text='+input
        //gets response and gets back the json, then we print out the json
      }).then((response) => response.json()).then((responseJson) => {
        brainrotParse(responseJson.result);
      }).catch((error) => {
        console.log(error);
        vidPlayer.play();
      });
}

function brainrotParse(data) {
    brainrot = data.Brainrot ;
    nonBrainrot = data.NonBrainrot ;
    //console.log(brainrot, nonBrainrot);
    const threshold = .70 ;
    //if brainrot value is bigger than brainrot then check to see if brainrot value
    //passes threshold. otherwise check nonbrainrot value against threshold
    if (brainrot > nonBrainrot){
        if (brainrot >= threshold){
            console.log(1) ;
            replaceVideo() ;
        }
    } else {
        console.log(0) ;
        vidPlayer.play();
    }
}

function replaceVideo(message = "This video is unproductive") {
    // Use Trusted Types to clear body content
    const policy = window.trustedTypes.createPolicy('default', {
        createHTML: (html) => html
    });

    // Clear the body content using TrustedHTML
    document.body.innerHTML = policy.createHTML('');

    // Create new elements
    const container = document.createElement('div');
    container.style.cssText = `
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 100vh;
        font-family: Arial, sans-serif;
        background-color: #f9f9f9;
        opacity: 0;
        transition: opacity 1s ease-in;
    `;

    const messageDiv = document.createElement('div');
    messageDiv.textContent = message;
    messageDiv.style.cssText = `
        font-size: 24px;
        margin-bottom: 20px;
    `;

    const redirectDiv = document.createElement('div');
    redirectDiv.textContent = "Redirecting to YouTube homepage in 5 seconds...";
    redirectDiv.style.cssText = `
        font-size: 18px;
    `;

    container.appendChild(messageDiv);
    container.appendChild(redirectDiv);
    document.body.appendChild(container);

    // Trigger reflow to ensure the transition works
    void container.offsetWidth;

    // Fade in the new content
    container.style.opacity = '1';

    // Set up redirection
    setTimeout(() => {
        window.location.href = 'https://www.youtube.com/';
    }, 5000);

    return true;
}
