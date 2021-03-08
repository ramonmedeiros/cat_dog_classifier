
async function previewFile() {
  var input = document.getElementById("myFile");
  var reader = new FileReader();
  var img1 = document.getElementById("animal");
  var predicted = document.getElementById("predicted");
  var model = await tf.loadGraphModel("https://ramonmedeiros.github.io/cat_dog_classifier/web_model/model.json");
  
  reader.onload = function(e) {  
    img1.src= e.target.result;
  }
  
  img1.onload = function() { 
    const imageTensor = tf.browser.fromPixels(img1);  
    const smalImg = tf.image.resizeBilinear(imageTensor, [150, 150]);
    const resized = tf.cast(smalImg, 'float32');
    const t4d = tf.tensor4d(Array.from(resized.dataSync()),[1,150,150,3])

    //const preprocessedInput = imageTensor.expandDims();
      
    const prediction = model.predict(t4d);
    predicted.innerHTML = prediction.shape;
    console.log(prediction);
  }

  if (input.files && input.files[0]) {
    reader.readAsDataURL(input.files[0]);
  } 
}
