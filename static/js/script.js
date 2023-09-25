const MODEL_URL = "model/model.json";

async function loadModel() {
  const model = await tf.loadLayersModel(MODEL_URL);
  return model;
}

const modelPromise = loadModel();

async function preprocessImage(imageElement) {
  const image = await tf.browser.fromPixels(imageElement);
  const resizedImage = tf.image.resizeBilinear(image, [224, 224]);
  const expandedImage = resizedImage.expandDims();
  const preprocessedImage = expandedImage.toFloat().div(tf.scalar(255));
  return preprocessedImage;
}

async function predictImage(model, preprocessedImage) {
  const prediction = await model.predict(preprocessedImage);
  return prediction;
}

async function displayResult(prediction) {
  const resultElement = document.getElementById("result");
  const predictedClass = tf.argMax(prediction, 1).dataSync()[0];
  const classNames = [
    "apple_pie",
    "baby_back_ribs",
    "baklava",
    "beef_carpaccio",
    "beef_tartare",
    "beet_salad",
    "beignets",
    "bibimbap",
    "bread_pudding",
    "breakfast_burrito",
    "bruschetta",
    "caesar_salad",
    "cannoli",
    "caprese_salad",
    "carrot_cake",
    "ceviche",
    "cheesecake",
    "cheese_plate",
    "chicken_curry",
    "chicken_quesadilla",
    "chicken_wings",
    "chocolate_cake",
    "chocolate_mousse",
    "churros",
    "clam_chowder",
    "club_sandwich",
    "crab_cakes",
    "creme_brulee",
    "croque_madame",
    "cup_cakes",
    "deviled_eggs",
    "donuts",
    "dumplings",
    "edamame",
    "eggs_benedict",
    "escargots",
    "falafel",
    "filet_mignon",
    "fish_and_chips",
    "foie_gras",
    "french_fries",
    "french_onion_soup",
    "french_toast",
    "fried_calamari",
    "fried_rice",
    "frozen_yogurt",
    "garlic_bread",
    "gnocchi",
    "greek_salad",
    "grilled_cheese_sandwich",
    "grilled_salmon",
    "guacamole",
    "gyoza",
    "hamburger",
    "hot_and_sour_soup",
    "hot_dog",
    "huevos_rancheros",
    "hummus",
    "ice_cream",
    "lasagna",
    "lobster_bisque",
    "lobster_roll_sandwich",
    "macaroni_and_cheese",
    "macarons",
    "miso_soup",
    "mussels",
    "nachos",
    "omelette",
    "onion_rings",
    "oysters",
    "pad_thai",
    "paella",
    "pancakes",
    "panna_cotta",
    "peking_duck",
    "pho",
    "pizza",
    "pork_chop",
    "poutine",
    "prime_rib",
    "pulled_pork_sandwich",
    "ramen",
    "ravioli",
    "red_velvet_cake",
    "risotto",
    "samosa",
    "sashimi",
    "scallops",
    "seaweed_salad",
    "shrimp_and_grits",
    "spaghetti_bolognese",
    "spaghetti_carbonara",
    "spring_rolls",
    "steak",
    "strawberry_shortcake",
    "sushi",
    "tacos",
    "takoyaki",
    "tiramisu",
    "tuna_tartare",
    "waffles",
  ];
  resultElement.innerText = `Predicted Class: ${classNames[predictedClass]}`;
}

async function handleUpload() {
  const uploadedImage = document.getElementById("fileInput");
  const file = uploadedImage.files[0];
  const reader = new FileReader();

  reader.onload = async function () {
    const img = new Image();
    img.src = reader.result;
    img.onload = async function () {
      const preprocessedImage = await preprocessImage(img);
      const model = await modelPromise;
      const prediction = await predictImage(model, preprocessedImage);
      await displayResult(prediction);
    };
  };

  if (file) {
    reader.readAsDataURL(file);
  }
}
