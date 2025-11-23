const sideMenu = document.querySelector('aside');
const menuBtn = document.querySelector('#menu_bar');
const closeBtn = document.querySelector('#close_btn');

menuBtn.addEventListener('click', () => {
    sideMenu.style.display = 'block';
});

closeBtn.addEventListener('click', () => {
    sideMenu.style.display = 'none';
});

const themeToggler = document.querySelector('.theme-toggler');

// Persist theme preference
try {
    const storedTheme = localStorage.getItem('theme-pref');
    if (storedTheme === 'dark') {
        document.body.classList.add('dark-theme-variables');
        themeToggler?.querySelector('span:nth-child(2)')?.classList.add('active');
    } else {
        themeToggler?.querySelector('span:nth-child(1)')?.classList.add('active');
    }
} catch (e) {}

themeToggler.addEventListener('click', () => {
    const isDark = document.body.classList.toggle('dark-theme-variables');
    themeToggler.querySelector('span:nth-child(1)').classList.toggle('active', !isDark);
    themeToggler.querySelector('span:nth-child(2)').classList.toggle('active', isDark);
    try { localStorage.setItem('theme-pref', isDark ? 'dark' : 'light'); } catch (e) {}
});

// Camera feature
const cameraIcons = document.querySelectorAll('.icon');
const cameraForms = document.querySelectorAll('.camera-form');

let currentFormIndex = -1;

cameraIcons.forEach((icon, index) => {
    icon.addEventListener('click', () => {
        if (currentFormIndex === index) {
            // Close the selected camera form
            cameraForms[index].style.display = 'none';
            currentFormIndex = -1;
        } else {
            // Close all camera forms
            cameraForms.forEach((form) => {
                form.style.display = 'none';
            });

            // Show the selected camera form
            cameraForms[index].style.display = 'block';
            currentFormIndex = index;
        }
    });
});

// Close camera forms when clicking outside
document.addEventListener('click', (event) => {
    if (!event.target.closest('.add-camera') && !event.target.closest('.camera-form')) {
        cameraForms.forEach((form) => {
            form.style.display = 'none';
        });
        currentFormIndex = -1;
    }
});

// Validate add camera FPS range
function validateCameraFPS(input) {
    const value = parseInt(input.value);
    if (isNaN(value) || value < 0 || value > 60) {
        input.setCustomValidity("Please enter a numeric value between 0 and 60.");
    } else {
        input.setCustomValidity("");
    }
}