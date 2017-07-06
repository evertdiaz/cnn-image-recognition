const requestify = require('requestify');
requestify.request('https://westcentralus.api.cognitive.microsoft.com/vision/v1.0/analyze', {
    method: 'POST',
    body: { 
      "url": "https://dl2.pushbulletusercontent.com/GCHZpIGXfgBf7owYsLMjsLKOpge1qHzk/20170622_153904.jpg" 
  },
    headers: {
      'Content-Type': 'application/json',
      'Ocp-Apim-Subscription-Key': 'TU_API_KEY'
    }
    })
    .then(function(response) {
      console.log(response.getBody())
        res.send(response.getBody());
    })
