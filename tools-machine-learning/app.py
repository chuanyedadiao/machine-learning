from flask import (Flask, render_template, g, flash, request, session, abort,
        redirect, url_for)
import DTClass

app = Flask(__name__)
dt = DTClass.DecisionTree()

@app.route('/')
def hello_world():  # put application's code here
    return redirect("display.html")

@app.route('/get_df/<filename>',methods=['GET'])
def get_df(filename):
    return str(dt.readData(filename=filename))

@app.route('/encoder',methods=['GET'])
def label_encoder():
    dt.labelEncoder()
    return str(dt.df)

@app.route('/splitXy/<y>',methods=['GET'])
def splitXy(y):
    dt.splitXy(y)
    return "split ok"

@app.route('/splitTrainTest',methods=["GET"])
def splitTrainTest():
    dt.splitTrainTest()
    return "split train test ok"

@app.route('/buildTree',methods=["GET"])
def buildTree():
    dt.buildTree()
    return "build OK"





if __name__ == '__main__':
    app.run()
