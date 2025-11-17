<?php
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $name = $_POST['name'];
    $email = $_POST['email'];
    $message = $_POST['message'];

    // Check if any of the form fields are empty
    if (empty($name) || empty($email) || empty($message)) {
        echo "Please fill in all fields.";
    } else {
        // Dummy email address to send the message
        $to = "dummy@example.com";
        $subject = "Message from Night Shield Contact Form";
        $body = "Name: $name\nEmail: $email\nMessage: $message";

        if (mail($to, $subject, $body)) {
            echo "Message sent successfully!";
        } else {
            echo "Error sending message.";
        }
    }
} else {
    echo "Invalid request.";
}
?>
