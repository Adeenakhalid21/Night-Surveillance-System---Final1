const sideMenu = document.querySelector('aside');
const menuBtn = document.querySelector('#menu_bar');
const closeBtn = document.querySelector('#close_btn');

if (menuBtn && sideMenu) {
    menuBtn.addEventListener('click', () => {
        sideMenu.style.display = 'block';
    });
}

if (closeBtn && sideMenu) {
    closeBtn.addEventListener('click', () => {
        sideMenu.style.display = 'none';
    });
}

const themeToggler = document.querySelector('.theme-toggler');
const lightIcon = themeToggler?.querySelector('span:nth-child(1)');
const darkIcon = themeToggler?.querySelector('span:nth-child(2)');

function applyTheme(theme) {
    const isDark = theme === 'dark';
    document.body.classList.toggle('dark-theme-variables', isDark);

    if (lightIcon && darkIcon) {
        lightIcon.classList.toggle('active', !isDark);
        darkIcon.classList.toggle('active', isDark);
    }

    try {
        localStorage.setItem('theme-pref', isDark ? 'dark' : 'light');
    } catch (e) {}
}

// Persist theme preference
try {
    const storedTheme = localStorage.getItem('theme-pref');
    if (storedTheme === 'dark' || storedTheme === 'light') {
        applyTheme(storedTheme);
    } else {
        applyTheme(document.body.classList.contains('dark-theme-variables') ? 'dark' : 'light');
    }
} catch (e) {}

if (themeToggler) {
    themeToggler.addEventListener('click', () => {
        applyTheme(document.body.classList.contains('dark-theme-variables') ? 'light' : 'dark');
    });
}

// Camera feature
const cameraIcons = document.querySelectorAll('.icon');
const cameraForms = document.querySelectorAll('.camera-form');

let currentFormIndex = -1;

cameraIcons.forEach((icon, index) => {
    icon.addEventListener('click', () => {
        if (!cameraForms[index]) {
            return;
        }

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