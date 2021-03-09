from flask import Flask, jsonify, request
from flask_cors import CORS
import intentclassifier
app = Flask(__name__)
CORS(app)

@app.route('/api/talk',methods=['POST'])
def index():
  user_input = request.json['user_input']
  iclf = intentclassifier.IntentClassifier()
  return jsonify({'msg':str(iclf.get_class(user_input))})
if __name__ == '__main__': 
  app.run()
