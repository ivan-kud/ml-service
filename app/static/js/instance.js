file = document.getElementById('fileBtn');
submit = document.getElementById('submitBtn');
output = document.getElementById('output');

function submitForm() {
    if (file.value == null || file.value == "") {
        output.innerHTML = 'Choose a file.';
    } else {
        output.innerHTML = 'Processing… Please wait up to 8 seconds.';
        document.getElementById('myForm').submit();
        file.disabled = true;
        submit.disabled = true;
    }
}
