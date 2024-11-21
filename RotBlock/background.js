chrome.runtime.onInstalled.addListener(() => {
    chrome.storage.local.set({rotblockActive: 'false'});
    console.log('Rotblock set to deactivated')
});

chrome.runtime.onStartup.addListener(() => {
    chrome.storage.local.get('rotblockActive', function(data) {
        if (data.rotblockActive === 'true' ){
            console.log('Rotblock filtering enabled on startup')
        } else {
            console.log('Rotblock filtering disabled on startup')
        }
    });
});

