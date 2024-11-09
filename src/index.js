import fs from 'fs';
import * as tf from "@tensorflow/tfjs-node";
import coco_ssd from "@tensorflow-models/coco-ssd";
import { Cam } from "onvif/promises/index.js"
import express from 'express';
import rtsp from "rtsp-ffmpeg";
import { config } from "dotenv";
config();

const app = express();
app.set('view engine', 'ejs');

let model = undefined;
let predictions = [];

(async () => {
  coco_ssd && coco_ssd.load({
    base: "mobilenet_v2",
  }).then((loadedModel) => {
    model = loadedModel;
    console.log("✅ Model Loaded")
  });
})();

const cam = new Cam({
  username: process.env.CAMERA_USER, 
  password: process.env.CAMERA_PASSWORD, 
  hostname: process.env.CAMERA_HOST, 
  port: process.env.CAMERA_PORT || 80
});

await cam.connect().then(() => {
  console.log('✅ Connected to camera');
});
const stream = await cam.getStreamUri({protocol:'RTSP'})

console.log('✅ Input from', stream.uri);

const video = new rtsp.FFMpeg({input: stream.uri, resolution: '640x480', quality: 8});
video.on('start', () => console.log('✅ Stream started'));
video.on('data', data => {
  fs.writeFileSync('./public/frame.png', Buffer.from(data, 'base64'));
});

video.on('exit', () => console.log('❌ stream exit'));

setInterval(async () => {
  try {
    const file = fs.readFileSync("./public/frame.png");
    const image = tf.node.decodeImage(file, 3);
    predictions = await model.detect(image, 3, 0.25);
  } catch (error) {
    if (!model) {
      console.log("❌ Model is not loaded yet!");
    } else {
      console.log("❌ Error", error);
    }
  }
}, 200);

app.get("/", (req, res) => {
  res.render('index', {
    predictions: predictions
  });
})

app.get("/health", (req, res) => {
  res.send("✅ Server is running");
});

app.use(express.static('public'))

app.get("/detect", async (req, res) => {
  res.json(predictions);
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`✅ Server started on port ${PORT}`);
});