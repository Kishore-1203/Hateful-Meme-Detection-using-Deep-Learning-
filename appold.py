from flask import Flask, render_template, flash, redirect, url_for, request, session, jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb

from sklearn.metrics import accuracy_score
import joblib
import pickle
from flask_mysqldb import MySQL
import secrets
# from flaskext.mysql import MySQL

app = Flask(__name__)
app.secret_key = 'your_secret_key'
# Initialize MySQL
mysql = MySQL(app)

# Configure MySQL
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = ''
app.config['MYSQL_DATABASE_DB'] = 'ids'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'  # Change this if your MySQL server is on a different host
app.config['MYSQL_DATABASE_AUTOCOMMIT'] = True

@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/dataupload')
def dataupload():
    return render_template('upload.html')

@app.route('/index')
def index():
    return render_template('index.html')

# Load the dataset
data = pd.read_csv('ids.csv')

# Convert object-type columns to numeric data types
for col in ['spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload']:
    unique_values = data[col].unique()
    print(f"Unique values in column '{col}': {unique_values}")

data['rate'] = pd.to_numeric(data['rate'], errors='coerce')
data.dropna(subset=['rate'], inplace=True)

data['spkts'] = pd.to_numeric(data['spkts'], errors='coerce')
data.dropna(subset=['spkts'], inplace=True)

data['sbytes'] = pd.to_numeric(data['sbytes'], errors='coerce')
data.dropna(subset=['sbytes'], inplace=True)

data['dbytes'] = pd.to_numeric(data['rate'], errors='coerce')
data.dropna(subset=['dbytes'], inplace=True)

data['dttl'] = pd.to_numeric(data['dttl'], errors='coerce')
data.dropna(subset=['dttl'], inplace=True)

data['dload'] = pd.to_numeric(data['dload'], errors='coerce')
data.dropna(subset=['dload'], inplace=True)

data['sload'] = pd.to_numeric(data['sload'], errors='coerce')
data.dropna(subset=['sload'], inplace=True)

data['sttl'] = pd.to_numeric(data['sttl'], errors='coerce')
data.dropna(subset=['sttl'], inplace=True)
# Drop rows with missing values
data.dropna(inplace=True)

# Encode the target variable
label_encoder = LabelEncoder()
data['attack_cat'] = label_encoder.fit_transform(data['attack_cat'])

# Split the dataset into features and target variable
X = data[['dur','spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload']]  # Features
y = data['attack_cat']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Features used in training:", X_train.columns.tolist())
# Save the trained model as a pickle file
with open('trained_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    print(model)

# Route for handling prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    dur = float(request.form['dur'])
    spkts = float(request.form['spkts'])
    dpkts = float(request.form['dpkts'])
    sbytes = float(request.form['sbytes'])
    dbytes = float(request.form['dbytes'])
    rate = float(request.form['rate'])
    sttl = float(request.form['sttl'])
    dttl = float(request.form['dttl'])
    sload = float(request.form['sload'])
    dload = float(request.form['dload'])

    # Create a DataFrame from the input data
    input_data = pd.DataFrame({
        'dur': [dur],
        'spkts': [spkts],
        'dpkts': [dpkts],
        'sbytes': [sbytes],
        'dbytes': [dbytes],
        'rate': [rate],
        'sttl': [sttl],
        'dttl': [dttl],
        'sload': [sload],
        'dload': [dload]
    })
    prediction = model.predict(input_data)[0]
    prediction_result = label_encoder.inverse_transform([prediction])[0]
    return redirect(url_for('prediction_result', result=prediction_result))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == 'admin' and password == 'admin':
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'danger')

    return render_template('login.html')

@app.route('/prediction_result')
def prediction_result():
    # Get the prediction result from the query parameter
    prediction_result = request.args.get('result')
    
    # Initialize remedies
    remedies = None
    
    # Provide remedies based on prediction result
    if prediction_result == "Dos":
        remedies = "Update your firewall settings and install intrusion detection systems."
    elif prediction_result == "Analysis":
        remedies = "Provide specific remedies for the other prediction result."
    elif prediction_result == "Backdoor":
        remedies = "Provide specific remedies for the other prediction result."
    elif prediction_result == "Exploits":
        remedies = "Provide specific remedies for the other prediction result."
    elif prediction_result == "Fuzzers":
        remedies = "Provide specific remedies for the other prediction result."
    elif prediction_result == "Generic":
        remedies = "Provide specific remedies for the other prediction result."
    elif prediction_result == "Reconnaissance":
        remedies = "Provide specific remedies for the other prediction result."
    elif prediction_result == "Shellcode":
        remedies = "Provide specific remedies for the other prediction result."
    elif prediction_result == "Worms":
        remedies = "Provide specific remedies for the other prediction result."
    else:
        remedies = "No specific remedies available."
    
    # Render the page with the prediction result and remedies
    return render_template('prediction_result.html', prediction_result=prediction_result, remedies=remedies)

@app.route('/insertdata', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        dur = float(request.form['dur'])
        spkts = float(request.form['spkts'])
        dpkts = float(request.form['dpkts'])
        sbytes = float(request.form['sbytes'])
        dbytes = float(request.form['dbytes'])
        rate = float(request.form['rate'])
        sttl = float(request.form['sttl'])
        dttl = float(request.form['dttl'])
        sload = float(request.form['sload'])
        dload = float(request.form['dload'])
       

        try:
            cursor = mysql.get_db().cursor()
            print("Cursor created successfully")
            cursor.execute("INSERT INTO prediction_data(dur, spkts, dpkts, sbytes,dbytes, rate,sttl,dttl,sload,dload) VALUES (%s, %s, %s, %s, %s)",
                            (dur, spkts, dpkts, sbytes, dbytes,rate,sttl,dttl,sload,dload))
            mysql.get_db().commit()
            print("Entered Data:",dur, spkts, dpkts, sbytes, dbytes,rate,sttl,dttl,sload,dload)
            print("Query executed successfully")
            print("Changes committed successfully")
        except Exception as e:
            print(f"Error inserting user: {e}")
        finally:
            if 'cursor' in locals():
                cursor.close()         
                 
        flash('Added successfully!', 'success')
        return redirect(url_for('attckdata'))
    return render_template('listdata.html')

def save_to_database(df):    
    cursor = mysql.get_db().cursor()
    print("Cursor created successfully")
    
    try:
        cursor.execute("TRUNCATE TABLE prediction_data")
        for index, row in df.iterrows():
            try:
                cursor.execute("INSERT INTO prediction_data (dur, spkts, dpkts, sbytes, dbytes, rate, sttl, dttl, sload, dload) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", (row['dur'], row['spkts'], row['dpkts'], row['sbytes'], row['dbytes'], row['rate'], row['sttl'], row['dttl'], row['sload'], row['dload']))
            except Exception as e:
                print(f"Error inserting user: {e}")
        mysql.get_db().commit()
        print("Data committed successfully")
    except Exception as e:
        print(f"Error saving data to database: {e}")
        mysql.get_db().rollback()
    finally:
        cursor.close()

@app.route('/upload',  methods=['GET', 'POST'])
def upload():
    try:
        file = request.files['file']
        if file:
            filename = file.filename
            if filename.endswith('.xlsx') or filename.endswith('.xls') or filename.endswith('.csv'):
                df = pd.read_excel(file) if filename.endswith('.xlsx') or filename.endswith('.xls') else pd.read_csv(file)
                save_to_database(df)
                flash('Data uploaded successfully!', 'success')
                return redirect('/success')
            else:
                flash('Please upload an Excel (xlsx, xls) or CSV file.', 'danger')
                return redirect('/')
        else:
            flash('No file uploaded.', 'danger')
            return redirect('/')
    except Exception as e:
        print("Error:", e)
        flash('An error occurred.', 'danger')
        return redirect('/')
    
@app.route('/get_prediction_data')
def get_prediction_data():
    try:
        id = request.args.get('id')
        cursor = mysql.get_db().cursor()
        cursor.execute("SELECT * FROM prediction_data WHERE id = %s ", (id))
        user = cursor.fetchone()
        cursor.close()
        if user:
            data = {
                'dur': user[1],
                'spkts': user[2],
                'dpkts': user[3],
                'sbytes': user[4],
                'dbytes': user[5],
                'rate': user[6],
                'sttl': user[7],
                'dttl': user[8],
                'sload': user[9],
                'dload': user[10]
            }
        else:
            data = {}

        return jsonify(data)      
    except Exception as e:
        print(f"Error fetching prediction data: {e}")
        return jsonify([])
       
@app.route('/success')
def success():
    return render_template('success.html')

if __name__ == "__main__":
    app.run(debug=True)
