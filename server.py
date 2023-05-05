from flask import Flask, jsonify, request
from flask_cors import CORS
import os, sys
import main_control_tower as mct
import clearFiles

terminal = "python3 /Users/jangjun-yeong/Pro-pose/main_control_tower.py"

app = Flask(__name__)
CORS(app)

@app.route('/auth/users', methods=['POST'])
def submit_data():
    data = request.get_json()  # retrieve data from the request body
    # do something with the data (e.g. save to a database)
    
    result = ''
    if __name__ == '__main__':
        result = mct.main(data['videoPath'], data['sort'], data['name'])
    clearFiles.clear(data['sort'], data['name'])
    # os.system(terminal)
    return {"result" : result}

if __name__ == '__main__' :
    app.run(debug = True)
    