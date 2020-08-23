const tf = require('@tensorflow/tfjs');
const mobilenet = require('@tensorflow-models/mobilenet');
const tfnode = require('@tensorflow/tfjs-node');
const fs = require('fs');


const readImage = path => {
  const imageBuffer = fs.readFileSync(path);
  const tfimage = tfnode.node.decodeImage(imageBuffer);
  return tfimage;
}

const imageClassification = async path => {
  const image = readImage(path);
  console.log('[info] loading the model...');
  const mobilenetModel = await mobilenet.load();
  console.log('[info] calculating results...');
  const predictions = await mobilenetModel.classify(image);
  console.log('Classification Resultes :', predictions);
}


if(process.argv.length!==3) 
  throw new Error('Incorrect argument');

imageClassification(process.argv[2]);
