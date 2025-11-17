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
    