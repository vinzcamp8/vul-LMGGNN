from flask import Flask, render_template, request, jsonify
import subprocess
import re
import threading

app = Flask(__name__, static_folder='templates/icons')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    code = request.form['code']
    with open('data/demo/100.c', 'w') as file:
        file.write(code)
    
    model_path = "data/model/gcn_0.0001_8_3_0.0001"
    sample_path = "data/demo/100.c"
    command = f"python3 demo.py --model_path={model_path} --sample_path={sample_path}"

    def run_subprocess():
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            result = f"Error: {stderr.decode('utf-8')}"
        else:
            decoded_stdout = stdout.decode('utf-8')
        
            # Parse the stdout to extract the required information
            time_to_generate_cpg = re.search(r'Time to generate CPG: (\d+\.\d+) seconds', decoded_stdout).group(1)
            time_to_embed = re.search(r'Time to embed: (\d+\.\d+) seconds', decoded_stdout).group(1)
            time_to_inference = re.search(r'Time to inference: (\d+\.\d+) seconds', decoded_stdout).group(1)
            predicted_label = re.search(r'Predicted label: \[\[(\d)\]\]', decoded_stdout).group(1)
            probability = re.search(r'Predicted probability: (\d\.\d+)', decoded_stdout).group(1)
            
            result = {
                'time_to_generate_cpg': time_to_generate_cpg,
                'time_to_embed': time_to_embed,
                'time_to_inference': time_to_inference,
                'predicted_label': predicted_label,
                'prediction_probability': probability
            }
        
        # Send the result back to the client
        app.config['RESULT'] = result

    # Run the subprocess in a separate thread
    thread = threading.Thread(target=run_subprocess)
    thread.start()

    return jsonify({'status': 'Processing...'}), 202

@app.route('/result', methods=['GET'])
def result():
    result = app.config.get('RESULT', None)
    if result:
        return jsonify(result)
    else:
        return jsonify({'status': 'Processing...'}), 202


if __name__ == '__main__':
    app.run(debug=True)