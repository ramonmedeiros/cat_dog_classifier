const WIDTH=150
const HEIGHT=150


async function previewFile() {
  var input = document.getElementById("myFile")
  var reader = new FileReader()
  var img1 = document.getElementById("animal")
  var predicted = document.getElementById("predicted")
  var model = await tf.loadGraphModel("https://ramonmedeiros.github.io/cat_dog_classifier/web_model/model.json")
  var resized = document.getElementById("resized")
  var canvas = document.getElementById("canva")


  reader.onload = function(e) {
    img1.src= e.target.result
  }

  img1.onload = function() {
    // set size proportional to image
    oc = canvas.getContext('2d');
    oc.width = WIDTH;
    oc.height = HEIGHT;
    oc.drawImage(img1, 0, 0, img1.width, img1.height, 0, 0, canvas.width, canvas.height);

    resized.width = WIDTH
    resized.height = HEIGHT
    resized.src = canvas.toDataURL("image/jpeg")
    

    var imageTensor = tf.browser.fromPixels(resized).toFloat();
    //imageTensor = imageTensor.expandDims()
    imageTensor = tf.reshape(imageTensor, [-1, WIDTH, HEIGHT, 3],'resize');
    var prediction = model.predict(imageTensor).dataSync()
    console.log(prediction)
    predicted.innerHTML = prediction
  }

  if (input.files && input.files[0]) {
    predicted.innerHTML = "Loading .."
    reader.readAsDataURL(input.files[0])
  }
}
