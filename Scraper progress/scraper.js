// tested using violentmonkey script runner chrome extension
// 

(function() {
    'use strict';

    const brainRotTags = [
        "shorts", "trending", "funny", "viral", "satisfying", "challenge",
        "fyp", "for you", "memes", "life hacks", "reaction", "vlog", "prank",
        "#memes", "#comedy", "#brainrot", "#addicting", "#bingeworthy", "#cantstopwatching", "#loop"
    ];

    const productiveTags = [
        "tutorial", "how to", "learning", "educational", "self-improvement",
        "coding", "tech", "motivation", "science", "documentary", "study",
        "#anime", "#manga", "#animation", "#music", "#gaming"
    ];

    let blockedVideos = [];
    let alertShown = false;

    function checkBrainRotReasons(title, description, tags, videoLength) {
        let reasons = [];
        const content = `${title} ${description} ${tags.join(' ')}`.toLowerCase();

        brainRotTags.forEach(tag => {
            if (content.includes(tag)) {
                reasons.push(`Contains unproductive tag: "${tag}"`);
            }
        });

        const videoMinutes = parseInt(videoLength.split(':')[0]) || 0;
        if (videoMinutes < 2) {
            reasons.push(`Video is too short (${videoLength})`);
        }

        productiveTags.forEach(tag => {
            if (content.includes(tag)) {
                reasons = [];  // Clear reasons if productive tag is found
            }
        });

        return reasons;
    }

    function removeBrainRotVideos() {
        const videos = document.querySelectorAll('ytd-rich-item-renderer, ytd-grid-video-renderer, ytd-video-renderer');

        console.log(`Found ${videos.length} videos on the page.`);

        videos.forEach(video => {
            const titleElement = video.querySelector('#video-title');
            const descriptionElement = video.querySelector('#description-text');
            const tagsElement = video.querySelector('#tags');
            const durationElement = video.querySelector('.ytd-thumbnail-overlay-time-status-renderer');

            const title = titleElement ? titleElement.innerText.trim() : '';
            const description = descriptionElement ? descriptionElement.innerText.trim() : '';
            const tags = tagsElement ? Array.from(tagsElement.querySelectorAll('a')).map(tag => tag.innerText.trim()) : [];
            const videoLength = durationElement ? durationElement.innerText.trim() : '0:00';

            console.log(`Checking video: ${title}, Length: ${videoLength}`);

            const reasons = checkBrainRotReasons(title, description, tags, videoLength);

            if (reasons.length > 0) {
                video.style.display = 'none';  // Hide the video
                if (!blockedVideos.some(v => v.title === title)) {
                    blockedVideos.push({ title: title, reasons: reasons });
                    console.log(`Blocked: ${title}, Reasons: ${reasons.join(', ')}`);
                }
            }
        });

        if (blockedVideos.length > 0 && !alertShown) {
            alertShown = true;
            showAlert();
        }

        // remove empty spaces due to mass video deleting
        removeEmptySpaces();
    }

    function showAlert() {
        const blockedDetails = blockedVideos.map(video =>
            `Title: ${video.title}\nReasons: ${video.reasons.join(', ')}`
        ).join('\n\n');

        if (blockedDetails) {
            alert('Blocked Unproductive Videos:\n\n' + blockedDetails);
        }

        blockedVideos = [];
    }

    function removeYouTubeShorts() {
        const shortsSelectors = [
            'a[href*="/shorts/"]',
            'ytd-grid-video-renderer[is-shorts]',
            'ytd-reel-shelf-renderer',
        ];

        shortsSelectors.forEach(selector => {
            const shortsElements = document.querySelectorAll(selector);
            shortsElements.forEach(element => {
                console.log('Removed a YouTube Shorts element:', element);
                element.remove(); // Remove the shorts element
            });
        });
    }

    function removeShortsTitleSpan() {
        const shortsTitleElements = document.querySelectorAll('span#title.style-scope.ytd-rich-shelf-renderer');
        shortsTitleElements.forEach(element => {
            console.log('Removed a Shorts title span:', element);
            element.remove();
        });
    }

    // Function to remove specified div elements
    function removeSpecifiedDivs() {
        const divsToRemove = document.querySelectorAll('div.yt-spec-touch-feedback-shape__fill');
        divsToRemove.forEach(div => {
            console.log('Removed specified div:', div);
            div.remove();
        });
    }

    // function to remove empty spaces by rearranging the layout
    function removeEmptySpaces() {
        const videoContainers = document.querySelectorAll('ytd-rich-item-renderer, ytd-grid-video-renderer, ytd-video-renderer');

        videoContainers.forEach(video => {
            if (video.style.display === 'none') {
                video.remove();  // completely remove the video element of each blocked vid
            }
        });

        const remainingVideos = document.querySelectorAll('ytd-rich-item-renderer:not([style*="display: none"]), ytd-grid-video-renderer:not([style*="display: none"]), ytd-video-renderer:not([style*="display: none"])');

        if (remainingVideos.length > 0) {
            remainingVideos.forEach(video => {
                // 
                video.style.marginBottom = '0';  
            });
        }
    }

    // Run the script when the page loads
    window.addEventListener('load', () => {
        removeBrainRotVideos();  // initial check
        removeYouTubeShorts();   // remove Shorts
        removeShortsTitleSpan(); // remove Shorts n title
        removeSpecifiedDivs();    // remove specified divs of shorts and show more buttons

        const observer = new MutationObserver(() => {
            removeBrainRotVideos();
            removeYouTubeShorts();
            removeShortsTitleSpan();
            removeSpecifiedDivs();
        });

        observer.observe(document.body, { childList: true, subtree: true });
    });
})();
