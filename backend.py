from flask import Flask, request, jsonify
import json
import subprocess

app = Flask(__name__)

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Netp(nn.Module): 
    def __init__(self):        
        super(Netp, self).__init__()        
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)    
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  
        self.fc2 = nn.Linear(120, 84)
        #self.fc3 = nn.Linear(84, 10)        

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = self.fc3(x)
#       output = F.softmax(x, dim=1)
        return x

def prepareInput(input):

  #get args

  pytorch_model = Netp()
  pytorch_model.load_state_dict(torch.load('./clientmodel.pt'))
  pytorch_model.eval()
  dummy_input = torch.reshape(torch.zeros(28 * 28 * 1),(1,1,28,28))
  #torch.onnx.export(pytorch_model, dummy_input, './components/clientmodel_bs16.onnx', verbose=True)
  """
  arr = []
  for i in range(0, 28):
    arr_ = []
    for j in range(0, 28):
      arr_.append(float(0))
    arr.append(arr_)
    
  a = torch.tensor(arr).reshape(1, 1, 28, 28)
  """
  # Example usage:
  # Assuming 'image' is a numpy array representing the input image
  #input = torch.reshape(torch.zeros(28 * 28 * 1),(1,1,28,28))

  prediction = pytorch_model.forward(input)
  prediction = prediction.detach().numpy().tolist()
  new = prediction
  print("Prediction:", new)

  new_ = []
  new_in = []
  for e in new[0]:
      new_in.append(int(e * 1000))
  new_.append(new_in)

  #convert new to json with key "data"

  new = {"image": [new_]*16}

  #write to file

  with open('./zk/input.json', 'w') as f:
    f.write(str(new).replace("\'", "\""))

  #print("Prediction shape:", prediction.shape)


# Configuration CORS pour autoriser les requêtes depuis tous les domaines
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

@app.route('/api/image', methods=['POST', 'OPTIONS'])
def receive_image():
    if request.method == 'OPTIONS':
        print("great")
        return '', 200  # Réponse vide pour la requête OPTIONS
    data = request.json
    #image = json.loads(data)
    #print(data)
    image = { "image": data.get('drawing') }

    arr = []
    for e in image["image"]:
        arr.append(e)

    for i in range(0, 28):
        for j in range(0, 28):
            #take floor of x scaled by 1000
            arr[i][j] = float(arr[i][j])

    print(arr)

    image_tensor = torch.tensor(arr).reshape(1, 1, 28, 28)
    prepareInput(image_tensor)

    #input_path = "./zk/input.json"
    #write image into the file
    #with open(input_path, 'w') as outfile:
        #json.dump(image, outfile)

    #Generate witness : node ./build/circuit_js/generate_witness.js ./build/circuit_js/circuit.wasm input.json witness.wtns
    subprocess.run(["node", "./zk/build/circuit_js/generate_witness.js", "./zk/build/circuit_js/circuit.wasm", "./zk/input.json", "./zk/build/witness.wtns"], cwd="./")

    #Generate proof "snarkjs groth16 prove ./build/circuit_0001.zkey ./build/witness.wtns proof.json public.json" with subprocess 
    subprocess.run(["snarkjs", "groth16", "prove", "./zk/build/circuit_0001.zkey", "./zk/build/witness.wtns", "./zk/proof.json", "./zk/public.json"], cwd="./")

    #get file ./proof.json and ./public.json
    with open("./zk/proof.json") as f:
        proof = json.load(f)

    with open("./zk/public.json") as f:
        public = json.load(f)

    #send proof and public to the client
        
    return jsonify({"proof": proof, "public": public})

    #print(image)
    # Faites ici le traitement de l'image (par exemple une prédiction)
    # Après traitement, renvoyez la réponse avec le tableau d'image modifié
    modified_image = [0, 1, 0, 1, 0, 1]  # Exemple de tableau d'image modifié
    return jsonify(modified_image)

if __name__ == '__main__':
    # Change the IP address to '0.0.0.0' to listen on all available network interfaces
    # Change the port number to the desired port, for example, 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
