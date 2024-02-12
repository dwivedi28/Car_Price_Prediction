from flask import Flask, request, render_template
from src.pipelines.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/form', methods=['GET', 'POST'])
def submit():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        carName = request.form['carName']
        year = int(request.form['year'])
        presentPrice = float(request.form['presentPrice'])
        kmsDriven = int(request.form['kmsDriven'])
        owner = int(request.form['owner'])
        fuelType = request.form['fuelType']
        sellerType = request.form['sellerType']
        transmissionType = request.form['transmissionType']

        data = CustomData(
            carName=carName,
            year=year,
            presentPrice=presentPrice,
            kmsDriven=kmsDriven,
            owner=owner,
            fuelType=fuelType,
            sellerType=sellerType,
            transmissionType=transmissionType
        )

        if fuelType == 'Diesel':
            Fuel_Type_Diesel = 1
            Fuel_Type_Petrol = 0
        else:
            Fuel_Type_Diesel = 0
            Fuel_Type_Petrol = 1

        if sellerType == 'Individual':
            Seller_Type_Individual = 1
        else:
            Seller_Type_Individual = 0
        
        if transmissionType == 'Manual':
            Transmission_Manual = 1
        else:
            Transmission_Manual = 0

        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        # Assuming you have a target_encoder object
        pred = predict_pipeline.predict(final_new_data)

        return render_template('form.html', price=pred)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
