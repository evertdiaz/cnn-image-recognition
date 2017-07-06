'use strict';

var VisualRecognitionV3 = require('watson-developer-cloud/visual-recognition/v3');
var fs = require('fs');

var visual_recognition = new VisualRecognitionV3({
  api_key: "TU_API_KEY",
  version_date: '2016-05-19'
});

var params = {
  // .zip contenedor de imagenes
  images_file: fs.createReadStream('./public/img/cachorro.zip')
};

visual_recognition.classify(params, function(err, res) {
  if (err)
    {console.log(err);}
  else
    {console.log(JSON.stringify(res, null, 2));}
});

