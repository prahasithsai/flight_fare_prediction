from flask import Flask,render_template,request
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def result():
    Total_Stops = request.form.get('Total_Stops')
    Journey_day = request.form.get('Journey_day')
    Journey_month = request.form.get('Journey_month')
    Dep_hour = request.form.get('Dep_hour')
    Dep_min = request.form.get('Dep_min')
    Arrival_hour = request.form.get('Arrival_hour')
    Arrival_min = request.form.get('Arrival_min')
    Duration_hour = request.form.get('Duration_hour')
    Duration_mins = request.form.get('Duration_mins')
    
    result = model.predict([[Total_Stops,Journey_day,Journey_month,Dep_hour,Dep_min,Arrival_hour,Arrival_min,Duration_hour,Duration_mins]])[0]

    return render_template('index.html', output='Flight Fare Price is {}'.format(result))

if __name__=='__main__':
    app.run(debug=False)
