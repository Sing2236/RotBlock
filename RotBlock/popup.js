// Get the toggle button element
const toggleButton = document.getElementById('toggleButton');

// Listen for button click
toggleButton.addEventListener('click', () => {
    // Check the current state of the button and toggle it
    if (toggleButton.innerText === 'ON') {
        toggleButton.innerText = 'OFF';
        console.log('Scraping activated');
        
        // Send a message to the content script to start scraping
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            chrome.tabs.sendMessage(tabs[0].id, { action: "startScraping" });
        });
    } else {
        toggleButton.innerText = 'ON';
    }
});
