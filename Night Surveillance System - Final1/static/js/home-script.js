function myMenuFunction() {
    var i = document.getElementById("navMenu");
    if(i.className === "nav-menu") {
        i.className += " responsive";
    } else {
        i.className = "nav-menu";
    }
   }
    var a = document.getElementById("loginBtn");
    var b = document.getElementById("registerBtn");
    var x = document.getElementById("login");
    var y = document.getElementById("register");
    
    function login() {
        x.style.left = "4px";
        y.style.right = "-20px";
        a.className += " white-btn";
        b.className = "btn";
        x.style.opacity = 1;
        y.style.opacity = 0;
    }
    function register() {
        x.style.left = "-510px";
        y.style.right = "5px";
        a.className = "btn";
        b.className += " white-btn";
        x.style.opacity = 0;
        y.style.opacity = 1;
    }

//function to handle user authentication and redirection
    function authenticateUser() {
        // For demonstration purposes, let's assume the username is "user" and password is "password"
        var username = document.getElementById("usernameField").value;
        var password = document.getElementById("passwordField").value;
    
        // Perform basic validation
        if (username === "demo" && password === "demo123") {
            // Redirect to dashboard upon successful login
            window.location.href = "/dashboard"; // Replace "dashboard.html" with your actual dashboard page
        } else {
            alert("Invalid username or password. Please try again."); // Display an error message for invalid credentials
        }
    }
    
// Show/Hide password toggles
document.addEventListener('DOMContentLoaded', function() {
    var loginPw = document.getElementById('passwordField');
    var regPw = document.getElementById('registerPasswordField');
    var tLogin = document.getElementById('toggleLoginPw');
    var tReg = document.getElementById('toggleRegisterPw');

    function wire(btn, input){
        if (!btn || !input) return;
        var icon = btn.querySelector('i');
        function setState(showing){
            if (!icon) return;
            icon.classList.toggle('bx-show', !showing);
            icon.classList.toggle('bx-hide', showing);
            btn.title = showing ? 'Hide password' : 'Show password';
            btn.setAttribute('aria-label', btn.title);
        }
        btn.addEventListener('click', function(){
            var isHidden = input.getAttribute('type') === 'password';
            input.setAttribute('type', isHidden ? 'text' : 'password');
            setState(isHidden);
        });
        // ensure initial icon state reflects current type
        setState(false);
    }

    wire(tLogin, loginPw);
    wire(tReg, regPw);
});
