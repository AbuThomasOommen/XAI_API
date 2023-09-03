document.getElementById('imageInput').addEventListener('change', function(event) {
    var file = event.target.files[0];
    var reader = new FileReader();
  
    reader.onload = function(e) {
      // document.getElementById("output_img").src = ""; // Clear previous image
      document.getElementById('selectedImage').src = e.target.result;
    };
    reader.readAsDataURL(file);
  });

// Add an event listener to the model selection dropdown
document.getElementById("modelSelect").addEventListener("change", function() {
//   var selectedModel = this.value;
  var selectedModelId = event.target.options[event.target.selectedIndex].getAttribute("data-id");
  
  // Determine which input method to show/hide based on the selected model
  if (selectedModelId === "100") {
    document.getElementById("imageContainer").style.display = "block";
    document.getElementById("tweetContainer").style.display = "none";
  } else if (selectedModelId === "200") {
    document.getElementById("tweetContainer").style.display = "block";
    document.getElementById("imageContainer").style.display = "none";
  } else {
    // Show neither input method if the default option is selected
    document.getElementById("imageContainer").style.display = "none";
    document.getElementById("tweetContainer").style.display = "none";
  }
});

  function runModel(event) {
    var selectedModel = document.getElementById("modelSelect").value;
    var selectedTweet = document.getElementById("tweetInput").value;
  
    // Check if the user has selected a model
    if (selectedModel === "Choose a model first") {
      alert("Please select a model before running the model.");
      return;
    }

    console.log("Selected Model is:", selectedModel);

    // var selectedModel = document.getElementById("modelSelect").value;
    document.getElementById("output_desc").textContent = selectedModel;
  
    // Show the loading message while waiting for the model to complete
    document.getElementById("loading").style.display = "block";

    // var selectedModelId = event.target.options[event.target.selectedIndex].getAttribute("data-id");
    var selectedModelId = document.getElementById("modelSelect").options[document.getElementById("modelSelect").selectedIndex].getAttribute("data-id");
    console.log("Selected Model ID is:", selectedModelId);

    if (selectedModelId === "100") 
    {
    var formData1 = new FormData();
    formData1.append('imageInput', document.getElementById("imageInput").files[0]);
  
    fetch('/predict', {method: 'POST',body: formData1})
      .then(response => response.json())
      .then(data => 
        {
        document.getElementById("output_msg").textContent = data.message;
        var formData2 = new FormData();
        formData2.append('imageInput', document.getElementById("imageInput").files[0]);
  
        fetch('/explain', {method: 'POST',body: formData2})
                .then(response => response.blob())
                .then(data => {
                        var imageUrl = URL.createObjectURL(data);                     
                        var outputImg = document.getElementById("output_img");
                        outputImg.onload = function() {
                                                    // Once the image is loaded, hide the loading message
                                                    document.getElementById("loading").style.display = "none";
                                                    outputImg.style.display = "block"; // Show the output image
                        };
      
                        outputImg.src = imageUrl;
                        })
                .catch(error => console.error('Explain Error:', error));
         })
      .catch(error => console.error('Predict Error:', error));
    }

    else if (selectedModelId === "200") 
    {
    var formData3 = new FormData();
    formData3.append('tweetInput', document.getElementById("tweetInput").value);
  
    fetch('/predict', {method: 'POST',body: formData3})
      .then(response => response.json())
      .then(data => 
        {
        document.getElementById("output_msg").textContent = data.message;
        var formData4 = new FormData();
        formData4.append('tweetInput', document.getElementById("tweetInput").value);
  
        fetch('/explain', {method: 'POST',body: formData4})
                .then(response => response.blob())
                .then(data => {
                        var imageUrl = URL.createObjectURL(data);                     
                        var outputImg = document.getElementById("output_img");
                        outputImg.onload = function() {
                                                    // Once the image is loaded, hide the loading message
                                                    document.getElementById("loading").style.display = "none";
                                                    outputImg.style.display = "block"; // Show the output image
                        };
      
                        outputImg.src = imageUrl;
                        })
                .catch(error => console.error('Explain Error:', error));
        })
      .catch(error => console.error('Predict Error:', error));
    }
  }