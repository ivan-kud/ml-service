text = document.getElementById('text');
output1 = document.getElementById('output1');
output2 = document.getElementById('output2');
submit = document.getElementById('submitBtn');

function submitForm() {
    if (text.value < 1) {
        output1.innerHTML = 'Write a review please.';
        output2.innerHTML = '&nbsp';
    }
    else {
        output1.innerHTML = 'Processing… Please wait.';
        output2.innerHTML = '&nbsp';
        document.getElementById('myForm').submit();
        submit.disabled = true;
    }
}
