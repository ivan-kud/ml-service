function submitForm() {
    document.getElementById('output').innerHTML = 'Processing… Please wait up to 8 seconds.';
    document.getElementById('myForm').submit();
    document.getElementById('submitBtn').disabled = true;
    document.getElementById('file').disabled = true;
}
