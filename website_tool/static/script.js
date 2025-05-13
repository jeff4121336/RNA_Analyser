// JavaScript to toggle visibility of input and upload boxes
document
  .getElementById("input-sequence-btn")
  .addEventListener("click", function () {
    document.getElementById("input-box").style.display = "block";
    document.getElementById("upload-box").style.display = "none";
    document.getElementById("input-sequence-btn").disabled = true;
    document.getElementById("upload-file-btn").disabled = false;
    document.getElementById("upload-method").value = "sequence";
    document.getElementById("result-box-muted").innerHTML = "";
  });

document
  .getElementById("upload-file-btn")
  .addEventListener("click", function () {
    document.getElementById("input-box").style.display = "none";
    document.getElementById("upload-box").style.display = "block";
    document.getElementById("upload-file-btn").disabled = true;
    document.getElementById("input-sequence-btn").disabled = false;
    document.getElementById("upload-method").value = "file";
    document.getElementById("result-box-muted").innerHTML = "";
  });

document
  .getElementById("submit-sequence")
  .addEventListener("click", function () {
    const form = document.getElementById("rna-form");
    const formData = new FormData(form);

    // Determine the upload method
    if (document.getElementById("user-input").value.trim() !== "") {
      formData.set("upload_method", "sequence");
    } else if (document.getElementById("file-upload").files.length > 0) {
      formData.set("upload_method", "file");
    } else {
      document.getElementById("result-box-muted").innerHTML =
        "<p id='result-box-muted' class='text-danger'>Please provide an RNA sequence or upload a file.</p>";
      return;
    }

    // Display loading message
    document.getElementById("result-box-muted").innerHTML =
      "<p id='result-box-muted' class='text-muted'>Processing...</p>";
  });
