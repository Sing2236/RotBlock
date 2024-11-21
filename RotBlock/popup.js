// Get the toggle button element
const toggleButton = document.getElementById('toggleButton');

chrome.storage.local.get('rotblockActive', function(data) {
    if (data.rotblockActive === 'true' ){
        toggleButton.innerText = 'Turn OFF'
    } else {
        toggleButton.innerText = 'Turn ON'
    }
});


// Listen for button click
toggleButton.addEventListener('click', () => {
    // Check the current state of the button and toggle it
    if (toggleButton.innerText === 'Turn ON') {
        chrome.storage.local.set({rotblockActive: 'true'});
        toggleButton.innerText = 'Turn OFF';
        console.log('Scraping activated');
        
        // Send a message to the content script to start scraping
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            chrome.tabs.sendMessage(tabs[0].id, { action: "startScraping" });
        });
    } else {
        chrome.storage.local.set({rotblockActive: 'false'});
        toggleButton.innerText = 'Turn ON';
    }
});
