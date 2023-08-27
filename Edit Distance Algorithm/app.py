from flask import Flask, request, render_template

app = Flask(__name__)

def edit_distance(str1, str2):
    m = len(str1)
    n = len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1])
    
    return dp[m][n]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    target_string = request.form['target_string']
    input_string = request.form['input_string']
    distance= edit_distance(target_string, input_string)
    return render_template('index.html',distance=distance)


if __name__ == '__main__':
    app.run(debug=True)
