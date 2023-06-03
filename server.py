from flask import Flask, render_template, request
import outPrediction as ml  

app = Flask(__name__)

@app.route('/', methods=['GET'])
def root():
    return render_template('index.html')

@app.route('/input', methods=["GET", "POST"])
def input():
    d =[]
    if request.method == 'GET':
        return render_template('input.html')
    elif request.method == 'POST':
        algorithm = int(request.form['Algorithm'])
        cocl = int(request.form['Coach class'])
        tot = int(request.form['Total Seats'])
        type = int(request.form['Type'])
        bks = int(request.form['Booking status'])
        max = int(request.form['Maximum wait list'])
        fest = int(request.form['fest'])
        hol = int(request.form['Holidays'])
        d.append(cocl)
        d.append(tot)
        d.append(type)
        d.append(bks)
        d.append(max)
        d.append(fest)
        d.append(hol)


        
        if(request.form['Status after 1 Week'].isdigit()):
            sa1w = int(request.form['Status after 1 Week'])
            d.append(sa1w)

        if(request.form['Status after 2 Week'].isdigit()):
            sa2w = int(request.form['Status after 2 Week'])
            d.append(sa2w)
        if(request.form['Status after 3 Week']):
            sa3w = int(request.form['Status after 3 Week'])
            d.append(sa3w)
        if(request.form['Status before 2 days']):
            sb2d = int(request.form['Status before 2 days'])
            d.append(sb2d)

        if(request.form['Status before 1 day']):
            sb1d=int(request.form['Status before 1 day'])
            d.append(sb1d)

        

        algorithm_name = ''
        accuracy = 0
        if algorithm == 0:
            prob,accuracy = ml.probability_generate(algorithm,[d])
            algorithm_name = 'GradientBoost'
        elif algorithm == 1:
            prob,accuracy = ml.probability_generate(algorithm,[d])
            algorithm_name = 'Ada Boost'
        elif algorithm == 2:
            prob,accuracy = ml.probability_generate(algorithm,[d])
            algorithm_name = 'Decision Tree'
        elif algorithm == 3:
            prob,accuracy = ml.probability_generate(algorithm,[d])
            algorithm_name = 'Random Forest'
        elif algorithm == 4:
            prob,accuracy = ml.probability_generate(algorithm,[d])
            algorithm_name = 'SVM'
        elif algorithm == 5:
            prob,accuracy = ml.probability_generate(algorithm,[d])
            algorithm_name = 'Naive Bayes'


        return render_template("result1.html", label=prob, accuracy=accuracy, algorithm=algorithm_name)




app.run(host='0.0.0.0', debug=False)