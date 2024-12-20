from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/update_parameters', methods=['POST'])
def update_parameters():
    data = request.get_json()
    k_value = data['k_value']
    iterations_value = data['iterations_value']
    eps_value = data['eps_value']
    minpts_value = data['minpts_value']
    
    return jsonify({
        'k_value': k_value,
        'iterations_value': iterations_value,
        'eps_value': eps_value,
        'minpts_value': minpts_value
    })

if __name__ == '__main__':
    app.run(debug=True)