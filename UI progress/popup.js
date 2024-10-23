document.addEventListener('DOMContentLoaded', () => {
    const toggleButton = document.getElementById('toggleButton');

    // Add a click event listener to toggle the button text
    toggleButton.addEventListener('click', () => {
        if (toggleButton.textContent === 'ON') {
            toggleButton.textContent = 'OFF';
        } else {
            toggleButton.textContent = 'ON';
        }
    });
});