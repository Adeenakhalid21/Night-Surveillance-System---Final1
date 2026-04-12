function setupTabs() {
    const loginTab = document.getElementById('loginTab');
    const registerTab = document.getElementById('registerTab');
    const loginPanel = document.getElementById('loginPanel');
    const registerPanel = document.getElementById('registerPanel');

    if (!loginTab || !registerTab || !loginPanel || !registerPanel) {
        return;
    }

    function showLogin() {
        loginTab.classList.add('active');
        registerTab.classList.remove('active');
        loginPanel.classList.remove('hidden');
        registerPanel.classList.add('hidden');
    }

    function showRegister() {
        registerTab.classList.add('active');
        loginTab.classList.remove('active');
        registerPanel.classList.remove('hidden');
        loginPanel.classList.add('hidden');
    }

    loginTab.addEventListener('click', showLogin);
    registerTab.addEventListener('click', showRegister);

    const firstRegisterInput = registerPanel.querySelector('input[name="firstname"]');
    if (firstRegisterInput && firstRegisterInput.value.trim()) {
        showRegister();
    } else {
        showLogin();
    }
}


window.myMenuFunction = function myMenuFunction() {
    const legacyNav = document.getElementById('navMenu');
    if (legacyNav && legacyNav.classList.contains('nav-menu')) {
        legacyNav.classList.toggle('responsive');
        return;
    }

    const modernNav = document.getElementById('siteNav');
    if (modernNav) {
        modernNav.classList.toggle('open');
    }
};


function setupPasswordToggles() {
    const toggles = document.querySelectorAll('.password-toggle');
    toggles.forEach((toggleBtn) => {
        const targetId = toggleBtn.getAttribute('data-target');
        const input = targetId ? document.getElementById(targetId) : null;
        const icon = toggleBtn.querySelector('i');
        if (!input || !icon) {
            return;
        }

        toggleBtn.addEventListener('click', () => {
            const isPassword = input.type === 'password';
            input.type = isPassword ? 'text' : 'password';
            icon.classList.toggle('bx-show', !isPassword);
            icon.classList.toggle('bx-hide', isPassword);
            toggleBtn.setAttribute('aria-label', isPassword ? 'Hide password' : 'Show password');
        });
    });
}


function setupMobileNav() {
    const menuBtn = document.getElementById('mobileMenuBtn');
    const nav = document.getElementById('siteNav');
    if (!menuBtn || !nav) {
        return;
    }

    menuBtn.addEventListener('click', () => {
        nav.classList.toggle('open');
    });
}


function setupParticles() {
    const canvas = document.getElementById('particleCanvas');
    if (!canvas) {
        return;
    }

    const ctx = canvas.getContext('2d');
    const particles = [];
    const particleCount = 56;

    function resizeCanvas() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }

    function createParticle() {
        return {
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            vx: (Math.random() - 0.5) * 0.35,
            vy: (Math.random() - 0.5) * 0.35,
            radius: Math.random() * 2 + 0.4,
            alpha: Math.random() * 0.45 + 0.15,
        };
    }

    function initParticles() {
        particles.length = 0;
        for (let i = 0; i < particleCount; i += 1) {
            particles.push(createParticle());
        }
    }

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        for (let i = 0; i < particles.length; i += 1) {
            const p = particles[i];

            p.x += p.vx;
            p.y += p.vy;

            if (p.x < 0 || p.x > canvas.width) {
                p.vx *= -1;
            }
            if (p.y < 0 || p.y > canvas.height) {
                p.vy *= -1;
            }

            ctx.beginPath();
            ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(0, 212, 255, ${p.alpha})`;
            ctx.fill();
        }

        requestAnimationFrame(draw);
    }

    resizeCanvas();
    initParticles();
    draw();
    window.addEventListener('resize', () => {
        resizeCanvas();
        initParticles();
    });
}


document.addEventListener('DOMContentLoaded', () => {
    setupTabs();
    setupPasswordToggles();
    setupMobileNav();
    setupParticles();
});
